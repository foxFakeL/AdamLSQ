#pragma once

#define NOMINMAX

#include "quant_pack.h"
#include "simd.h"
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <torch/extension.h>

#define TILE (128 * 1024 * 1024)

// Per-group LSQ quantization parameters
struct LSQ_Params {
  float *delta;   // scale parameter (per-group)
  float *z;       // zero point parameter (per-group)
  int group_size; // number of elements per group (e.g., 128)
  int q_bits;     // quantization bits (4 or 8)
};

class Adam_LSQ_Optimizer {
public:
  Adam_LSQ_Optimizer(float alpha = 1e-3, float betta1 = 0.9,
                     float betta2 = 0.999, float eps = 1e-8,
                     float weight_decay = 0, bool adamw_mode = true)
      : _alpha(alpha), _betta1(betta1), _betta2(betta2), _eps(eps),
        _weight_decay(weight_decay), _betta1_t(1.0), _betta2_t(1.0), _step(0),
        _adamw_mode(adamw_mode), _delta_ptr(nullptr), _z_ptr(nullptr),
        _group_size(128), _q_bits(8) {}
  ~Adam_LSQ_Optimizer() {}

  // Register quantization parameters (delta and z pointers)
  // Called once during initialization, stored internally for all subsequent
  // steps
  void register_quant_params(float *delta, float *z, int group_size,
                             int q_bits) {
    _delta_ptr = delta;
    _z_ptr = z;
    _group_size = group_size;
    _q_bits = q_bits;
  }

  // Getter methods for registered quantization parameters
  float *get_delta_ptr() const { return _delta_ptr; }
  float *get_z_ptr() const { return _z_ptr; }
  int get_group_size() const { return _group_size; }
  int get_q_bits() const { return _q_bits; }

// SIMD implementation of fused Adam + LSQ (AVX512/AVX2/NEON)
#if defined(__AVX512__) || defined(__AVX256__) || defined(__aarch64__)
  template <int span, typename ds_params_precision_t,
            typename ds_state_precision_t>
  void Step_AVX_LSQ(size_t *rounded_size, ds_params_precision_t *_params,
                    ds_params_precision_t *grads,
                    ds_state_precision_t *_exp_avg,
                    ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                    LSQ_Params &lsq_params, size_t _param_size);
#endif

  // Step functions for scalar fallback
  template <typename ds_params_precision_t, typename ds_state_precision_t>
  void Step_1_LSQ(ds_params_precision_t *_params, ds_params_precision_t *grads,
                  ds_state_precision_t *_exp_avg,
                  ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                  LSQ_Params &lsq_params, size_t _param_size);

  template <typename ds_params_precision_t, typename ds_state_precision_t>
  void Step_4_LSQ(ds_params_precision_t *_params, ds_params_precision_t *grads,
                  ds_state_precision_t *_exp_avg,
                  ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                  LSQ_Params &lsq_params, size_t _param_size);

  template <typename ds_params_precision_t, typename ds_state_precision_t>
  void Step_8_LSQ(ds_params_precision_t *_params, ds_params_precision_t *grads,
                  ds_state_precision_t *_exp_avg,
                  ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                  LSQ_Params &lsq_params, size_t _param_size);

  inline void IncrementStep(size_t step, float beta1, float beta2) {
    if (beta1 != _betta1 || beta2 != _betta2) {
      _step = step;
      _betta1 = beta1;
      _betta2 = beta2;
      _betta1_t = std::pow(_betta1, step);
      _betta2_t = std::pow(_betta2, step);
    } else {
      _step++;
      if (_step != step) {
        _betta1_t = std::pow(_betta1, step);
        _betta2_t = std::pow(_betta2, step);
        _step = step;
      } else {
        _betta1_t *= _betta1;
        _betta2_t *= _betta2;
      }
    }
  }

  inline void update_state(float lr, float epsilon, float weight_decay,
                           bool bias_correction) {
    _alpha = lr;
    _eps = epsilon;
    _weight_decay = weight_decay;

    _bias_correction1 = 1.0f;
    _bias_correction2 = 1.0f;
    if (bias_correction == 1) {
      _bias_correction1 = 1 - _betta1_t;
      _bias_correction2 = 1 / sqrt(1 - _betta2_t);
    }
  }

private:
  float _alpha;
  float _betta1;
  float _betta2;
  float _eps;
  float _weight_decay;

  float _betta1_t;
  float _betta2_t;
  size_t _step;

  float _bias_correction1;
  float _bias_correction2;

  bool _adamw_mode;

  // Internal storage for quantization parameters (registered once during init)
  float *_delta_ptr;
  float *_z_ptr;
  int _group_size;
  int _q_bits;
};

// SIMD implementation of fused Adam + LSQ (AVX512/AVX2/NEON)
#if defined(__AVX512__) || defined(__AVX256__) || defined(__aarch64__)
template <int span, typename ds_params_precision_t,
          typename ds_state_precision_t>
void Adam_LSQ_Optimizer::Step_AVX_LSQ(
    size_t *rounded_size, ds_params_precision_t *_params,
    ds_params_precision_t *grads, ds_state_precision_t *_exp_avg,
    ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
    LSQ_Params &lsq_params, size_t _param_size) {

#if !defined(__AVX512__) && !defined(__aarch64__)
  if (std::is_same_v<ds_params_precision_t, c10::BFloat16> ||
      std::is_same_v<ds_state_precision_t, c10::BFloat16>) {
    // AVX2 doesn't support BF16 well, fall back to scalar
    Step_1_LSQ(_params, grads, _exp_avg, _exp_avg_sq, _quant_data, lsq_params,
               _param_size);
    *rounded_size = 0;
    return;
  }
#endif

  size_t new_rounded_size = 0;
  int q_max = lsq_params.q_bits == 4 ? 15 : 255;

  AVX_Data betta1_4;
  betta1_4.data = SIMD_SET(_betta1);
  AVX_Data betta2_4;
  betta2_4.data = SIMD_SET(_betta2);
  AVX_Data betta1_minus1_4;
  betta1_minus1_4.data = SIMD_SET(1 - _betta1);
  AVX_Data betta2_minus1_4;
  betta2_minus1_4.data = SIMD_SET(1 - _betta2);
  AVX_Data bias2_sqrt;
  bias2_sqrt.data = SIMD_SET(_bias_correction2);
  AVX_Data eps_4;
  eps_4.data = SIMD_SET(_eps);
  AVX_Data step_size_4;
  step_size_4.data = SIMD_SET(-1 * _alpha / _bias_correction1);

  AVX_Data weight_decay4;
  if (_weight_decay > 0)
    weight_decay4.data = (_adamw_mode ? SIMD_SET(-1 * _alpha * _weight_decay)
                                      : SIMD_SET(_weight_decay));

  int group_size = lsq_params.group_size;
  int simd_batch_size = SIMD_WIDTH * span;
  new_rounded_size = ROUND_DOWN(_param_size, simd_batch_size);

  // === TILE-based OpenMP调度（保留缓存局部性）===
  size_t num_tiles =
      new_rounded_size / TILE + (new_rounded_size % TILE ? 1 : 0);

#pragma omp parallel for schedule(static)
  for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
    size_t tile_start = tile_idx * TILE;
    size_t tile_end = tile_start + TILE;
    if (tile_end > new_rounded_size)
      tile_end = new_rounded_size;

    // === 核心优化1：消除紧凑循环内的除法！使用边界累加判断 ===
    int group_id = tile_start / group_size;
    size_t next_group_boundary = (size_t)(group_id + 1) * group_size;

    float delta_g = lsq_params.delta[group_id];
    float z_g = lsq_params.z[group_id];

#if defined(__aarch64__)
    float32x4_t z_v = vdupq_n_f32(z_g);
    float32x4_t delta_inv_v = vdupq_n_f32(1.0f / delta_g);
    float32x4_t zero_v = vdupq_n_f32(0.0f);
    float32x4_t max_v =
        (lsq_params.q_bits == 8) ? vdupq_n_f32(255.0f) : vdupq_n_f32(15.0f);
#elif defined(__AVX512__)
    AVX_Data z_v, delta_inv_v, zero_v, max_v_local;
    z_v.data = SIMD_SET(z_g);
    delta_inv_v.data = SIMD_SET(1.0f / delta_g);
    zero_v.data = SIMD_SET(0.0f);
    max_v_local.data = SIMD_SET((float)q_max);
#endif

    for (size_t i = tile_start; i < tile_end; i += simd_batch_size) {
      // === 核心优化1续：使用分支预测绕过硬件除法指令 ===
      // __builtin_expect(0) 表示这个分支极少发生（每group_size次才发生一次）
      if (__builtin_expect(i >= next_group_boundary, 0)) {
        group_id++;
        next_group_boundary += group_size;
        delta_g = lsq_params.delta[group_id];
        z_g = lsq_params.z[group_id];
#if defined(__aarch64__)
        z_v = vdupq_n_f32(z_g);
        delta_inv_v = vdupq_n_f32(1.0f / delta_g);
#elif defined(__AVX512__)
        z_v.data = SIMD_SET(z_g);
        delta_inv_v.data = SIMD_SET(1.0f / delta_g);
#endif
      }

      // --- DeepSpeed 原生 Adam ---
      AVX_Data grad_4[span], momentum_4[span], variance_4[span], param_4[span];
      simd_load<span>(grad_4, grads + i);
      simd_load<span>(momentum_4, _exp_avg + i);
      simd_load<span>(variance_4, _exp_avg_sq + i);
      simd_load<span>(param_4, _params + i);

      if (_weight_decay > 0 && !_adamw_mode)
        simd_fma<span>(grad_4, param_4, weight_decay4, grad_4);
      simd_mul<span>(momentum_4, momentum_4, betta1_4);
      simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
      simd_mul<span>(variance_4, variance_4, betta2_4);
      simd_mul<span>(grad_4, grad_4, grad_4);
      simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);
      simd_sqrt<span>(grad_4, variance_4);
      simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
      simd_div<span>(grad_4, momentum_4, grad_4);
      if (_weight_decay > 0 && _adamw_mode)
        simd_fma<span>(param_4, param_4, weight_decay4, param_4);
      simd_fma<span>(param_4, grad_4, step_size_4, param_4);

      simd_store<span>(_params + i, param_4);
      simd_store<span>(_exp_avg + i, momentum_4);
      simd_store<span>(_exp_avg_sq + i, variance_4);

      // === LSQ Quantization ===
#if defined(__aarch64__)
      if (lsq_params.q_bits == 8) {
        // NEON INT8: Process 4 vectors at a time to avoid register spilling
        for (int s = 0; s < span; s += 4) {
          int32x4_t q_int[4];
          for (int step = 0; step < 4; step++) {
            float32x4_t vals = param_4[s + step].data;
            float32x4_t sub = vsubq_f32(vals, z_v);
            float32x4_t scaled = vmulq_f32(sub, delta_inv_v);
            float32x4_t rounded = vrndnq_f32(scaled);
            float32x4_t clamped = vmaxq_f32(zero_v, vminq_f32(rounded, max_v));
            q_int[step] = vcvtq_s32_f32(clamped);
          }
          uint16x4_t q16_0 = vmovn_u32(vreinterpretq_u32_s32(q_int[0]));
          uint16x4_t q16_1 = vmovn_u32(vreinterpretq_u32_s32(q_int[1]));
          uint16x4_t q16_2 = vmovn_u32(vreinterpretq_u32_s32(q_int[2]));
          uint16x4_t q16_3 = vmovn_u32(vreinterpretq_u32_s32(q_int[3]));
          uint16x8_t q16_01 = vcombine_u16(q16_0, q16_1);
          uint16x8_t q16_23 = vcombine_u16(q16_2, q16_3);
          uint8x8_t q8_01 = vmovn_u16(q16_01);
          uint8x8_t q8_23 = vmovn_u16(q16_23);
          uint8x16_t final_q8 = vcombine_u8(q8_01, q8_23);

          uint64_t* dst64 = (uint64_t*)(_quant_data + i + s * SIMD_WIDTH);
          uint64x2_t data64 = vreinterpretq_u64_u8(final_q8);
          asm volatile("stnp %0, %1, [%2]" : : "r"(vgetq_lane_u64(data64, 0)), "r"(vgetq_lane_u64(data64, 1)), "r"(dst64) : "memory");
        }
      } else {
        // NEON INT4: 每次处理4向量(16元素)，打包成8字节
        // 和INT8一样用循环，避免寄存器溢出
        for (int s = 0; s < span; s += 4) {
          int32x4_t q_int[4];
          for (int step = 0; step < 4; step++) {
            float32x4_t vals = param_4[s + step].data;
            float32x4_t sub = vsubq_f32(vals, z_v);
            float32x4_t scaled = vmulq_f32(sub, delta_inv_v);
            float32x4_t rounded = vrndnq_f32(scaled);
            float32x4_t clamped = vmaxq_f32(zero_v, vminq_f32(rounded, max_v));
            q_int[step] = vcvtq_s32_f32(clamped);
          }
          // 压缩到8位
          uint16x4_t q16_0 = vmovn_u32(vreinterpretq_u32_s32(q_int[0]));
          uint16x4_t q16_1 = vmovn_u32(vreinterpretq_u32_s32(q_int[1]));
          uint16x4_t q16_2 = vmovn_u32(vreinterpretq_u32_s32(q_int[2]));
          uint16x4_t q16_3 = vmovn_u32(vreinterpretq_u32_s32(q_int[3]));
          uint16x8_t q16_01 = vcombine_u16(q16_0, q16_1);
          uint16x8_t q16_23 = vcombine_u16(q16_2, q16_3);
          uint8x8_t q8_01 = vmovn_u16(q16_01);
          uint8x8_t q8_23 = vmovn_u16(q16_23);
          uint8x16_t q8_0123 = vcombine_u8(q8_01, q8_23);

          // INT4打包：偶数位<<4 | 奇数位
          // q8_0123包含16个8位值，需要打包成8字节
          // 用vuzp分离偶数位和奇数位
          uint8x8_t evens = vuzp1_u8(vget_low_u8(q8_0123), vget_high_u8(q8_0123));
          uint8x8_t odds = vuzp2_u8(vget_low_u8(q8_0123), vget_high_u8(q8_0123));
          uint8x8_t packed = vorr_u8(vshl_n_u8(evens, 4), odds);

          uint64_t* dst64 = (uint64_t*)(_quant_data + (i + s * SIMD_WIDTH) / 2);
          uint64x1_t data64 = vreinterpret_u64_u8(packed);
          asm volatile("str %0, [%1]" : : "r"(vget_lane_u64(data64, 0)), "r"(dst64) : "memory");
        }
      }
#elif defined(__AVX512__)
      if (lsq_params.q_bits == 8) {
        for (int s = 0; s < span; s++) {
          AVX_Data sub_v, scaled_v, rounded_v, clamped_v;
          sub_v.data = SIMD_SUB(param_4[s].data, z_v.data);
          scaled_v.data = SIMD_MUL(sub_v.data, delta_inv_v.data);
          rounded_v.data = SIMD_ROUND(scaled_v.data);
          clamped_v.data =
              SIMD_MAX(zero_v.data, SIMD_MIN(rounded_v.data, max_v_local.data));
          __m512i q_int = _mm512_cvttps_epi32(clamped_v.data);
          __m128i q_8 = _mm512_cvtepi32_epi8(q_int);
          // 非时序存储：避免RFO写惩罚，绕过Cache直接写入内存
          _mm_stream_si128((__m128i *)(_quant_data + i + s * SIMD_WIDTH), q_8);
        }
      } else {
        for (int s = 0; s < span; s += 2) {
          AVX_Data sub_v0, scaled_v0, rounded_v0, clamped_v0;
          sub_v0.data = SIMD_SUB(param_4[s].data, z_v.data);
          scaled_v0.data = SIMD_MUL(sub_v0.data, delta_inv_v.data);
          rounded_v0.data = SIMD_ROUND(scaled_v0.data);
          clamped_v0.data = SIMD_MAX(
              zero_v.data, SIMD_MIN(rounded_v0.data, max_v_local.data));
          __m512i q_int0 = _mm512_cvttps_epi32(clamped_v0.data);
          __m128i q0_8 = _mm512_cvtepi32_epi8(q_int0);

          AVX_Data sub_v1, scaled_v1, rounded_v1, clamped_v1;
          sub_v1.data = SIMD_SUB(param_4[s + 1].data, z_v.data);
          scaled_v1.data = SIMD_MUL(sub_v1.data, delta_inv_v.data);
          rounded_v1.data = SIMD_ROUND(scaled_v1.data);
          clamped_v1.data = SIMD_MAX(
              zero_v.data, SIMD_MIN(rounded_v1.data, max_v_local.data));
          __m512i q_int1 = _mm512_cvttps_epi32(clamped_v1.data);
          __m128i q1_8 = _mm512_cvtepi32_epi8(q_int1);

          __m128i q0_shifted = _mm_mullo_epi16(q0_8, _mm_set1_epi16(16));
          __m128i packed_int4 = _mm_or_si128(q0_shifted, q1_8);
          // 非时序存储：避免RFO写惩罚
          // 提取低64位写入（INT4打包后是8字节）
          _mm_stream_si64((long long *)(_quant_data + (i + s * SIMD_WIDTH) / 2),
                           _mm_cvtsi128_si64(packed_int4));
        }
      }
#else
      // Scalar fallback: SIMD_WIDTH=1, span=8
      // INT8可以并行，INT4需要串行处理相邻元素对
      if (lsq_params.q_bits == 8) {
#pragma omp parallel for schedule(static)
        for (int s = 0; s < span; s++) {
          float val = (float)param_4[s].data;  // SIMD_WIDTH=1, 直接取float
          float q = (val - z_g) / delta_g;
          q = round_int(q);
          q = clamp_quant(q, 0.0f, 255.0f);
          _quant_data[i + s] = (uint8_t)q;  // s*SIMD_WIDTH = s*1 = s
        }
      } else {
        // INT4: 串行处理相邻元素对（避免数据竞争）
        // span=8, 处理(s, s+1)打包成1字节
        for (int s = 0; s < span; s += 2) {
          float val0 = (float)param_4[s].data;
          float q0 = (val0 - z_g) / delta_g;
          q0 = clamp_quant(round_int(q0), 0.0f, 15.0f);

          float q1 = 0.0f;
          if (s + 1 < span) {
            float val1 = (float)param_4[s + 1].data;
            q1 = (val1 - z_g) / delta_g;
            q1 = clamp_quant(round_int(q1), 0.0f, 15.0f);
          }

          // 打包：(q0 << 4) | q1
          _quant_data[(i + s) / 2] = ((uint8_t)((int)q0) << 4) | (uint8_t)((int)q1);
        }
      }
#endif
    }
  }
  *rounded_size = new_rounded_size;
}
#endif

// API functions
int create_adam_lsq_optimizer(int optimizer_id, float alpha = 1e-3,
                              float betta1 = 0.9, float betta2 = 0.999,
                              float eps = 1e-8, float weight_decay = 0,
                              bool adamw_mode = true, bool should_log = false);

// Register quantization parameters (delta and z) - called once per parameter
// during initialization
int register_quant_params_lsq(int optimizer_id, torch::Tensor &delta,
                              torch::Tensor &z, int group_size, int q_bits);

// Step function - uses internally stored delta/z, only passes 5 tensors
// (reduced from 7)
int ds_adam_step_lsq(int optimizer_id, size_t step, float lr, float beta1,
                     float beta2, float epsilon, float weight_decay,
                     bool bias_correction, torch::Tensor &params,
                     torch::Tensor &grads, torch::Tensor &exp_avg,
                     torch::Tensor &exp_avg_sq, torch::Tensor &quant_data);

int destroy_adam_lsq_optimizer(int optimizer_id);