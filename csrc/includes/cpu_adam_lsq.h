#pragma once

#define NOMINMAX

#include "quant_pack.h"
#include "simd.h"
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <torch/extension.h>

#define TILE (128 * 1024 * 1024)

class Adam_LSQ_Optimizer {
public:
  Adam_LSQ_Optimizer(float alpha = 1e-3, float betta1 = 0.9,
                     float betta2 = 0.999, float eps = 1e-8,
                     float weight_decay = 0, bool adamw_mode = true)
      : _alpha(alpha), _betta1(betta1), _betta2(betta2), _eps(eps),
        _weight_decay(weight_decay), _betta1_t(1.0), _betta2_t(1.0), _step(0),
        _adamw_mode(adamw_mode) {}
  ~Adam_LSQ_Optimizer() {}

  // 无状态架构：不再存储delta/z指针
  // 所有量化参数每次step传入

// SIMD implementation of fused Adam + LSQ (AVX512/AVX2/NEON)
#if defined(__AVX512F__) || defined(__AVX256__) || defined(__aarch64__)
  template <int span, typename ds_params_precision_t,
            typename ds_state_precision_t>
  void Step_AVX_LSQ(size_t *rounded_size, ds_params_precision_t *_params,
                    ds_params_precision_t *grads,
                    ds_state_precision_t *_exp_avg,
                    ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                    float *delta, float *z, int group_size, int q_bits,
                    size_t _param_size);
#endif

  // Step functions for scalar fallback
  template <typename ds_params_precision_t, typename ds_state_precision_t>
  void Step_1_LSQ(ds_params_precision_t *_params, ds_params_precision_t *grads,
                  ds_state_precision_t *_exp_avg,
                  ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                  float *delta, float *z, int group_size, int q_bits,
                  size_t _param_size);

  template <typename ds_params_precision_t, typename ds_state_precision_t>
  void Step_4_LSQ(ds_params_precision_t *_params, ds_params_precision_t *grads,
                  ds_state_precision_t *_exp_avg,
                  ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                  float *delta, float *z, int group_size, int q_bits,
                  size_t _param_size);

  template <typename ds_params_precision_t, typename ds_state_precision_t>
  void Step_8_LSQ(ds_params_precision_t *_params, ds_params_precision_t *grads,
                  ds_state_precision_t *_exp_avg,
                  ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
                  float *delta, float *z, int group_size, int q_bits,
                  size_t _param_size);

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

  // 无状态架构：不存储量化参数
};

// SIMD implementation of fused Adam + LSQ (AVX512/AVX2/NEON)
// OMP parallel在元素级别（DeepSpeed模式）
#if defined(__AVX512F__) || defined(__AVX256__) || defined(__aarch64__)
template <int span, typename ds_params_precision_t,
          typename ds_state_precision_t>
void Adam_LSQ_Optimizer::Step_AVX_LSQ(
    size_t *rounded_size, ds_params_precision_t *_params,
    ds_params_precision_t *grads, ds_state_precision_t *_exp_avg,
    ds_state_precision_t *_exp_avg_sq, uint8_t *_quant_data,
    float *delta, float *z, int group_size, int q_bits,
    size_t _param_size) {

#if !defined(__AVX512F__) && !defined(__aarch64__)
  if (std::is_same_v<ds_params_precision_t, c10::BFloat16> ||
      std::is_same_v<ds_state_precision_t, c10::BFloat16>) {
    // AVX2 doesn't support BF16 well, fall back to scalar
    Step_1_LSQ(_params, grads, _exp_avg, _exp_avg_sq, _quant_data,
               delta, z, group_size, q_bits, _param_size);
    *rounded_size = 0;
    return;
  }
#endif

  size_t new_rounded_size = 0;
  int q_max = q_bits == 4 ? 15 : 255;

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

  int simd_batch_size = SIMD_WIDTH * span;
  new_rounded_size = ROUND_DOWN(_param_size, simd_batch_size);

  // === DeepSpeed模式：外层TILE串行，内层元素OMP parallel ===
  for (size_t t = 0; t < new_rounded_size; t += TILE) {
    size_t tile_end = t + TILE;
    if (tile_end > new_rounded_size)
      tile_end = new_rounded_size;

#pragma omp parallel for schedule(static)
    for (size_t i = t; i < tile_end; i += simd_batch_size) {
      // 每个线程独立计算group_id（避免跨线程状态共享）
      int group_id_local = i / group_size;
      size_t next_group_boundary = (size_t)(group_id_local + 1) * group_size;

      float delta_g = delta[group_id_local];
      float z_g = z[group_id_local];

#if defined(__aarch64__)
      float32x4_t z_v = vdupq_n_f32(z_g);
      float32x4_t delta_inv_v = vdupq_n_f32(1.0f / delta_g);
      float32x4_t zero_v = vdupq_n_f32(0.0f);
      float32x4_t max_v = (q_bits == 8) ? vdupq_n_f32(255.0f) : vdupq_n_f32(15.0f);
#elif defined(__AVX512F__)
      AVX_Data z_v, delta_inv_v, zero_v, max_v_local;
      z_v.data = SIMD_SET(z_g);
      delta_inv_v.data = SIMD_SET(1.0f / delta_g);
      zero_v.data = SIMD_SET(0.0f);
      max_v_local.data = SIMD_SET((float)q_max);
#endif

      // === 循环内部：处理simd_batch_size内的元素 ===
      // 由于simd_batch_size通常小于group_size，大多数情况下不需要切换group
      for (size_t j = i; j < i + simd_batch_size && j < tile_end; j += SIMD_WIDTH * span) {
        // 检查是否需要切换group（simd_batch_size可能跨group边界）
        if (j >= next_group_boundary) {
          group_id_local = j / group_size;
          next_group_boundary = (size_t)(group_id_local + 1) * group_size;
          delta_g = delta[group_id_local];
          z_g = z[group_id_local];
#if defined(__aarch64__)
          z_v = vdupq_n_f32(z_g);
          delta_inv_v = vdupq_n_f32(1.0f / delta_g);
#elif defined(__AVX512F__)
          z_v.data = SIMD_SET(z_g);
          delta_inv_v.data = SIMD_SET(1.0f / delta_g);
#endif
        }

        // --- DeepSpeed 原生 Adam ---
        AVX_Data grad_4[span], momentum_4[span], variance_4[span], param_4[span];
        simd_load<span>(grad_4, grads + j);
        simd_load<span>(momentum_4, _exp_avg + j);
        simd_load<span>(variance_4, _exp_avg_sq + j);
        simd_load<span>(param_4, _params + j);

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

        simd_store<span>(_params + j, param_4);
        simd_store<span>(_exp_avg + j, momentum_4);
        simd_store<span>(_exp_avg_sq + j, variance_4);

        // === LSQ Quantization (skip if _quant_data is nullptr) ===
        if (_quant_data == nullptr) {
          continue;
        }

#if defined(__aarch64__)
        if (q_bits == 8) {
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

            uint64_t* dst64 = (uint64_t*)(_quant_data + j + s * SIMD_WIDTH);
            uint64x2_t data64 = vreinterpretq_u64_u8(final_q8);
            vst1q_u64(dst64, data64);
          }
        } else {
          // NEON INT4: 每次处理4向量(16元素)，打包成8字节
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
            uint8x16_t q8_0123 = vcombine_u8(q8_01, q8_23);

            // INT4打包：偶数位<<4 | 奇数位
            uint8x8_t evens = vuzp1_u8(vget_low_u8(q8_0123), vget_high_u8(q8_0123));
            uint8x8_t odds = vuzp2_u8(vget_low_u8(q8_0123), vget_high_u8(q8_0123));
            uint8x8_t packed = vorr_u8(vshl_n_u8(evens, 4), odds);

            uint64_t* dst64 = (uint64_t*)(_quant_data + (j + s * SIMD_WIDTH) / 2);
            vst1_u64(dst64, vreinterpret_u64_u8(packed));
          }
        }
#elif defined(__AVX512F__)
        if (q_bits == 8) {
          for (int s = 0; s < span; s++) {
            AVX_Data sub_v, scaled_v, rounded_v, clamped_v;
            sub_v.data = SIMD_SUB(param_4[s].data, z_v.data);
            scaled_v.data = SIMD_MUL(sub_v.data, delta_inv_v.data);
            rounded_v.data = SIMD_ROUND(scaled_v.data);
            clamped_v.data = SIMD_MAX(zero_v.data, SIMD_MIN(rounded_v.data, max_v_local.data));
            __m512i q_int = _mm512_cvttps_epi32(clamped_v.data);
            __m128i q_8 = _mm512_cvtepi32_epi8(q_int);
            _mm_stream_si128((__m128i *)(_quant_data + j + s * SIMD_WIDTH), q_8);
          }
        } else {
          for (int s = 0; s < span; s += 2) {
            AVX_Data sub_v0, scaled_v0, rounded_v0, clamped_v0;
            sub_v0.data = SIMD_SUB(param_4[s].data, z_v.data);
            scaled_v0.data = SIMD_MUL(sub_v0.data, delta_inv_v.data);
            rounded_v0.data = SIMD_ROUND(scaled_v0.data);
            clamped_v0.data = SIMD_MAX(zero_v.data, SIMD_MIN(rounded_v0.data, max_v_local.data));
            __m512i q_int0 = _mm512_cvttps_epi32(clamped_v0.data);
            __m128i q0_8 = _mm512_cvtepi32_epi8(q_int0);

            AVX_Data sub_v1, scaled_v1, rounded_v1, clamped_v1;
            sub_v1.data = SIMD_SUB(param_4[s + 1].data, z_v.data);
            scaled_v1.data = SIMD_MUL(sub_v1.data, delta_inv_v.data);
            rounded_v1.data = SIMD_ROUND(scaled_v1.data);
            clamped_v1.data = SIMD_MAX(zero_v.data, SIMD_MIN(rounded_v1.data, max_v_local.data));
            __m512i q_int1 = _mm512_cvttps_epi32(clamped_v1.data);
            __m128i q1_8 = _mm512_cvtepi32_epi8(q_int1);

            __m128i q0_shifted = _mm_mullo_epi16(q0_8, _mm_set1_epi16(16));
            __m128i packed_int4 = _mm_or_si128(q0_shifted, q1_8);
            _mm_stream_si64((long long *)(_quant_data + (j + s * SIMD_WIDTH) / 2),
                             _mm_cvtsi128_si64(packed_int4));
          }
        }
#endif
      }
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

// Step function with LSQ quantization - 无状态架构：每次传入所有参数
int ds_adam_step_lsq(int optimizer_id, size_t step, float lr, float beta1,
                     float beta2, float epsilon, float weight_decay,
                     bool bias_correction, torch::Tensor &params,
                     torch::Tensor &grads, torch::Tensor &exp_avg,
                     torch::Tensor &exp_avg_sq, torch::Tensor &quant_data,
                     torch::Tensor &delta, torch::Tensor &z,
                     int group_size, int q_bits);

// Step function without quantization - only Adam update
int ds_adam_step(int optimizer_id, size_t step, float lr, float beta1,
                 float beta2, float epsilon, float weight_decay,
                 bool bias_correction, torch::Tensor &params,
                 torch::Tensor &grads, torch::Tensor &exp_avg,
                 torch::Tensor &exp_avg_sq);

int destroy_adam_lsq_optimizer(int optimizer_id);