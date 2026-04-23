#pragma once

#include <stdint.h>

// INT4/INT8 packing helper functions

// Clamp value to range [min_val, max_val]
inline float clamp_quant(float val, float min_val, float max_val) {
  return val < min_val ? min_val : (val > max_val ? max_val : val);
}

// Round to nearest integer
inline int round_int(float val) { return (int)(val + 0.5f); }

// INT8 packing: single bf16/fp32 value -> uint8
// q = clamp(round((x - z) / delta), 0, 255)
inline uint8_t pack_int8_single(float val, float delta, float z) {
  float q = (val - z) / delta;
  q = clamp_quant(round_int(q), 0.0f, 255.0f);
  return (uint8_t)q;
}

// INT4 packing: two bf16/fp32 values -> one uint8
// High 4 bits store val1, Low 4 bits store val2
// q = clamp(round((x - z) / delta), 0, 15)
inline uint8_t pack_int4_pair(float val1, float val2, float delta, float z) {
  float q1 = (val1 - z) / delta;
  float q2 = (val2 - z) / delta;
  int q1_int = (int)clamp_quant(round_int(q1), 0.0f, 15.0f);
  int q2_int = (int)clamp_quant(round_int(q2), 0.0f, 15.0f);
  return ((uint8_t)q1_int << 4) | (uint8_t)q2_int;
}

// Unpack INT4 to float values (for dequantization/testing)
inline void unpack_int4_pair(uint8_t packed, float *val1, float *val2,
                             float delta, float z) {
  int q1 = (packed >> 4) & 0x0F;
  int q2 = packed & 0x0F;
  *val1 = q1 * delta + z;
  *val2 = q2 * delta + z;
}

// Unpack INT8 to float value (for dequantization/testing)
inline float unpack_int8_single(uint8_t packed, float delta, float z) {
  return packed * delta + z;
}