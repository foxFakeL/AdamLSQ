#include "cpu_adam_lsq.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <torch/extension.h>
#include <type_traits>
#include <unordered_map>

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers_lsq;

// Scalar fallback implementation for Step_1_LSQ
template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_LSQ_Optimizer::Step_1_LSQ(
    ds_params_precision_t *_params, ds_params_precision_t *grads,
    ds_state_precision_t *_exp_avg, ds_state_precision_t *_exp_avg_sq,
    uint8_t *_quant_data, float *delta, float *z, int group_size, int q_bits,
    size_t _param_size) {
  size_t rounded_size = 0;
#if defined(__AVX512F__) || defined(__AVX256__) || defined(__aarch64__)
  Step_AVX_LSQ<1>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq,
                  _quant_data, delta, z, group_size, q_bits, _param_size);
#endif

  if (_param_size > rounded_size) {
    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    float step_size = -1 * _alpha / _bias_correction1;
    float w_decay = -1 * _alpha * _weight_decay;
    int q_max = q_bits == 4 ? 15 : 255;

    // === 正确性修复：INT4需要串行处理相邻元素对，避免数据竞争 ===
    // INT8: 可以按元素并行（每个元素独立量化）
    // INT4: 必须按元素对串行处理（k和k+1必须在同一线程内完成Adam+量化）
    size_t stride = (q_bits == 4) ? 2 : 1;

    for (size_t t = rounded_size; t < _param_size; t += TILE) {
      size_t copy_size = TILE;
      if ((t + TILE) > _param_size)
        copy_size = _param_size - t;
      size_t offset = copy_size + t;

      // INT8: parallel for (safe)
      // INT4: serial loop (k and k+1 must be processed together)
      if (q_bits == 8) {
#pragma omp parallel for schedule(static)
        for (size_t k = t; k < offset; k++) {
          float grad = (float)grads[k];
          float param = (float)_params[k];
          float momentum = _exp_avg[k];
          float variance = _exp_avg_sq[k];

          if (_weight_decay > 0 && !_adamw_mode) {
            grad = param * _weight_decay + grad;
          }
          momentum = momentum * _betta1;
          momentum = grad * betta1_minus1 + momentum;
          variance = variance * _betta2;
          grad = grad * grad;
          variance = grad * betta2_minus1 + variance;
          grad = sqrt(variance);
          grad = grad * _bias_correction2 + _eps;
          grad = momentum / grad;
          if (_weight_decay > 0 && _adamw_mode) {
            param += w_decay * param;
          }
          param = grad * step_size + param;

          _params[k] = param;
          _exp_avg[k] = momentum;
          _exp_avg_sq[k] = variance;

          int group_id = k / group_size;
          float q = (param - z[group_id]) / delta[group_id];
          q = round_int(q);
          q = clamp_quant(q, 0.0f, (float)q_max);
          _quant_data[k] = (uint8_t)q;
        }
      } else {
        // INT4: Serial processing of element pairs (k, k+1)
        // No parallel for - must process adjacent elements together
        for (size_t k = t; k < offset; k += 2) {
          // Process element k
          float grad0 = (float)grads[k];
          float param0 = (float)_params[k];
          float momentum0 = _exp_avg[k];
          float variance0 = _exp_avg_sq[k];

          if (_weight_decay > 0 && !_adamw_mode) {
            grad0 = param0 * _weight_decay + grad0;
          }
          momentum0 = momentum0 * _betta1;
          momentum0 = grad0 * betta1_minus1 + momentum0;
          variance0 = variance0 * _betta2;
          grad0 = grad0 * grad0;
          variance0 = grad0 * betta2_minus1 + variance0;
          grad0 = sqrt(variance0);
          grad0 = grad0 * _bias_correction2 + _eps;
          grad0 = momentum0 / grad0;
          if (_weight_decay > 0 && _adamw_mode) {
            param0 += w_decay * param0;
          }
          param0 = grad0 * step_size + param0;

          _params[k] = param0;
          _exp_avg[k] = momentum0;
          _exp_avg_sq[k] = variance0;

          int group_id0 = k / group_size;
          float q0 = (param0 - z[group_id0]) / delta[group_id0];
          q0 = round_int(q0);
          q0 = clamp_quant(q0, 0.0f, 15.0f);

          // Process element k+1 (if exists)
          if (k + 1 < _param_size) {
            float grad1 = (float)grads[k + 1];
            float param1 = (float)_params[k + 1];
            float momentum1 = _exp_avg[k + 1];
            float variance1 = _exp_avg_sq[k + 1];

            if (_weight_decay > 0 && !_adamw_mode) {
              grad1 = param1 * _weight_decay + grad1;
            }
            momentum1 = momentum1 * _betta1;
            momentum1 = grad1 * betta1_minus1 + momentum1;
            variance1 = variance1 * _betta2;
            grad1 = grad1 * grad1;
            variance1 = grad1 * betta2_minus1 + variance1;
            grad1 = sqrt(variance1);
            grad1 = grad1 * _bias_correction2 + _eps;
            grad1 = momentum1 / grad1;
            if (_weight_decay > 0 && _adamw_mode) {
              param1 += w_decay * param1;
            }
            param1 = grad1 * step_size + param1;

            _params[k + 1] = param1;
            _exp_avg[k + 1] = momentum1;
            _exp_avg_sq[k + 1] = variance1;

            int group_id1 = (k + 1) / group_size;
            float q1 = (param1 - z[group_id1]) / delta[group_id1];
            q1 = round_int(q1);
            q1 = clamp_quant(q1, 0.0f, 15.0f);

            // Pack: (q0 << 4) | q1
            _quant_data[k / 2] = ((uint8_t)((int)q0) << 4) | (uint8_t)((int)q1);
          } else {
            // Edge case: odd number of elements, last element packed with 0
            _quant_data[k / 2] = ((uint8_t)((int)q0) << 4);
          }
        }
      }
    }
  }
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_LSQ_Optimizer::Step_4_LSQ(
    ds_params_precision_t *_params, ds_params_precision_t *grads,
    ds_state_precision_t *_exp_avg, ds_state_precision_t *_exp_avg_sq,
    uint8_t *_quant_data, float *delta, float *z, int group_size, int q_bits,
    size_t _param_size) {
  size_t rounded_size = 0;
#if defined(__AVX512F__) || defined(__AVX256__) || defined(__aarch64__)
  Step_AVX_LSQ<4>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq,
                  _quant_data, delta, z, group_size, q_bits, _param_size);
#endif
  if (_param_size > rounded_size) {
    Step_1_LSQ((_params + rounded_size), (grads + rounded_size),
               (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
               (_quant_data + (q_bits == 4 ? rounded_size / 2 : rounded_size)),
               delta, z, group_size, q_bits, (_param_size - rounded_size));
  }
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_LSQ_Optimizer::Step_8_LSQ(
    ds_params_precision_t *_params, ds_params_precision_t *grads,
    ds_state_precision_t *_exp_avg, ds_state_precision_t *_exp_avg_sq,
    uint8_t *_quant_data, float *delta, float *z, int group_size, int q_bits,
    size_t _param_size) {
  size_t rounded_size = 0;
#if defined(__AVX512F__) || defined(__AVX256__) || defined(__aarch64__)
  Step_AVX_LSQ<8>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq,
                  _quant_data, delta, z, group_size, q_bits, _param_size);
#endif
  if (_param_size > rounded_size) {
    Step_4_LSQ((_params + rounded_size), (grads + rounded_size),
               (_exp_avg + rounded_size), (_exp_avg_sq + rounded_size),
               (_quant_data + (q_bits == 4 ? rounded_size / 2 : rounded_size)),
               delta, z, group_size, q_bits, (_param_size - rounded_size));
  }
}

inline void invoke_lsq_direct(std::shared_ptr<Adam_LSQ_Optimizer> opt,
                              void *params, void *grads, void *exp_avg,
                              void *exp_avg_sq, uint8_t *quant_data,
                              float *delta, float *z, int group_size, int q_bits,
                              size_t param_size,
                              c10::ScalarType params_type,
                              c10::ScalarType state_type) {
  switch (params_type) {
    case c10::ScalarType::BFloat16:
      if (state_type == c10::ScalarType::Float) {
        // BF16 params + FP32 state
        opt->Step_8_LSQ((c10::BFloat16 *)params, (c10::BFloat16 *)grads,
                        (float *)exp_avg, (float *)exp_avg_sq, quant_data,
                        delta, z, group_size, q_bits, param_size);
      } else if (state_type == c10::ScalarType::BFloat16) {
        // BF16 params + BF16 state
        opt->Step_8_LSQ((c10::BFloat16 *)params, (c10::BFloat16 *)grads,
                        (c10::BFloat16 *)exp_avg, (c10::BFloat16 *)exp_avg_sq,
                        quant_data, delta, z, group_size, q_bits, param_size);
      } else {
        throw std::runtime_error("Unsupported state type for BF16 params");
      }
      break;
    case c10::ScalarType::Float:
      // FP32 params + FP32 state
      opt->Step_8_LSQ((float *)params, (float *)grads, (float *)exp_avg,
                      (float *)exp_avg_sq, quant_data,
                      delta, z, group_size, q_bits, param_size);
      break;
    default:
      throw std::runtime_error("Unsupported params type");
  }
}

int create_adam_lsq_optimizer(int optimizer_id, float alpha, float betta1,
                              float betta2, float eps, float weight_decay,
                              bool adamw_mode, bool should_log) {
  auto opt = std::make_shared<Adam_LSQ_Optimizer>(alpha, betta1, betta2, eps,
                                                  weight_decay, adamw_mode);

  s_optimizers_lsq[optimizer_id] = opt;

  return 0;
}

// Main step function with LSQ - 无状态架构：每次传入所有参数
int ds_adam_step_lsq(int optimizer_id, size_t step, float lr, float beta1,
                     float beta2, float epsilon, float weight_decay,
                     bool bias_correction, torch::Tensor &params,
                     torch::Tensor &grads, torch::Tensor &exp_avg,
                     torch::Tensor &exp_avg_sq, torch::Tensor &quant_data,
                     torch::Tensor &delta, torch::Tensor &z,
                     int group_size, int q_bits) {
  std::shared_ptr<Adam_LSQ_Optimizer> opt =
      std::static_pointer_cast<Adam_LSQ_Optimizer>(
          s_optimizers_lsq[optimizer_id]);

  opt->IncrementStep(step, beta1, beta2);
  opt->update_state(lr, epsilon, weight_decay, bias_correction);

  c10::ScalarType params_type = at::typeMetaToScalarType(params.options().dtype());
  c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg.options().dtype());

  invoke_lsq_direct(opt, params.data_ptr(), grads.data_ptr(), exp_avg.data_ptr(),
                    exp_avg_sq.data_ptr(), quant_data.data_ptr<uint8_t>(),
                    delta.data_ptr<float>(), z.data_ptr<float>(),
                    group_size, q_bits, params.numel(), params_type, state_type);

  return 0;
}

// Step function without quantization - only Adam update
int ds_adam_step(int optimizer_id, size_t step, float lr, float beta1,
                 float beta2, float epsilon, float weight_decay,
                 bool bias_correction, torch::Tensor &params,
                 torch::Tensor &grads, torch::Tensor &exp_avg,
                 torch::Tensor &exp_avg_sq) {
  std::shared_ptr<Adam_LSQ_Optimizer> opt =
      std::static_pointer_cast<Adam_LSQ_Optimizer>(
          s_optimizers_lsq[optimizer_id]);

  opt->IncrementStep(step, beta1, beta2);
  opt->update_state(lr, epsilon, weight_decay, bias_correction);

  c10::ScalarType params_type = at::typeMetaToScalarType(params.options().dtype());
  c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg.options().dtype());

  // 直接调用Adam更新，不创建dummy buffer
  // 使用单元素dummy delta/z，避免大规模内存分配
  float dummy_delta = 1.0f;
  float dummy_z = 0.0f;

  // 使用param_size作为group_size（所有元素在同一个组）
  // 这样只需要单元素的delta/z，避免创建大数组
  size_t param_size = params.numel();

  switch (params_type) {
    case c10::ScalarType::BFloat16:
      if (state_type == c10::ScalarType::Float) {
        // 调用内部Adam-only更新（不量化）
        opt->Step_8_LSQ((c10::BFloat16 *)params.data_ptr(),
                        (c10::BFloat16 *)grads.data_ptr(),
                        (float *)exp_avg.data_ptr(),
                        (float *)exp_avg_sq.data_ptr(),
                        nullptr,  // quant_data不写入
                        &dummy_delta, &dummy_z,
                        param_size, 8, param_size);
      } else {
        opt->Step_8_LSQ((c10::BFloat16 *)params.data_ptr(),
                        (c10::BFloat16 *)grads.data_ptr(),
                        (c10::BFloat16 *)exp_avg.data_ptr(),
                        (c10::BFloat16 *)exp_avg_sq.data_ptr(),
                        nullptr, &dummy_delta, &dummy_z,
                        param_size, 8, param_size);
      }
      break;
    case c10::ScalarType::Float:
      opt->Step_8_LSQ((float *)params.data_ptr(),
                      (float *)grads.data_ptr(),
                      (float *)exp_avg.data_ptr(),
                      (float *)exp_avg_sq.data_ptr(),
                      nullptr, &dummy_delta, &dummy_z,
                      param_size, 8, param_size);
      break;
    default:
      throw std::runtime_error("Unsupported params type");
  }

  return 0;
}

int destroy_adam_lsq_optimizer(int optimizer_id) {
  s_optimizers_lsq.erase(optimizer_id);
  return 0;
}