#pragma once

#include <stdint.h>
#include <torch/extension.h>
#include <cstring>
#include <type_traits>
#include <cmath>

#define TILE (128 * 1024 * 1024)
#define ROUND_DOWN(size, step) ((size) & ~((step) - 1))

//=============================================================================
// x86 AVX512/AVX2 Support
//=============================================================================
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

#if defined(__AVX512F__) || defined(__AVX256__)
#include <immintrin.h>

template <typename T>
inline T readAs(const void* src) {
    T res;
    std::memcpy(&res, src, sizeof(T));
    return res;
}
template <typename T>
inline void writeAs(void* dst, const T& val) {
    std::memcpy(dst, &val, sizeof(T));
}

#if defined(__AVX512F__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_ADD(x, y) _mm512_add_ps(x, y)
#define SIMD_SUB(x, y) _mm512_sub_ps(x, y)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_MAX(x, y) _mm512_max_ps(x, y)
#define SIMD_MIN(x, y) _mm512_min_ps(x, y)
#define SIMD_ROUND(x) _mm512_round_ps(x, _MM_FROUND_NINT)
#define SIMD_WIDTH 16

#elif defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_ADD(x, y) _mm256_add_ps(x, y)
#define SIMD_SUB(x, y) _mm256_sub_ps(x, y)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_MAX(x, y) _mm256_max_ps(x, y)
#define SIMD_MIN(x, y) _mm256_min_ps(x, y)
#define SIMD_ROUND(x) _mm256_round_ps(x, _MM_FROUND_NINT)
#define SIMD_WIDTH 8
#endif

union AVX_Data {
#if defined(__AVX512F__)
    __m512 data;
#elif defined(__AVX256__)
    __m256 data;
#endif
};

// x86 simd_load/store templates
template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_store(T* dst, AVX_Data* src) {
    for (size_t i = 0; i < span; ++i) { SIMD_STORE(dst + SIMD_WIDTH * i, src[i].data); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> simd_store(T* dst, AVX_Data* src) {
    for (size_t i = 0; i < span; ++i) {
        float tmp[SIMD_WIDTH];
        SIMD_STORE(tmp, src[i].data);
        for (int j = 0; j < SIMD_WIDTH; j++) { dst[SIMD_WIDTH * i + j] = c10::BFloat16(tmp[j]); }
    }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_load(AVX_Data* dst, T* src) {
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_LOAD(src + SIMD_WIDTH * i); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> simd_load(AVX_Data* dst, T* src) {
    float tmp[SIMD_WIDTH];
    for (size_t i = 0; i < span; ++i) {
        for (int j = 0; j < SIMD_WIDTH; j++) { tmp[j] = (float)src[SIMD_WIDTH * i + j]; }
        dst[i].data = SIMD_LOAD(tmp);
    }
}

// x86 SIMD arithmetic operations
template <int span> inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data* src_a) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a[i].data); } }
template <int span> inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data src_a) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a.data); } }
template <int span> inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r[i].data); } }
template <int span> inline void simd_div(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_DIV(src_a_l[i].data, src_a_r[i].data); } }
template <int span> inline void simd_sqrt(AVX_Data* dst, AVX_Data* src) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_SQRT(src[i].data); } }
template <int span> inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r[i].data); } }
template <int span> inline void simd_sub(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_SUB(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_max(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MAX(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_min(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MIN(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_round(AVX_Data* dst, AVX_Data* src) { for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ROUND(src[i].data); } }

#endif // AVX512 or AVX256

//=============================================================================
// ARM NEON Support (aarch64)
//=============================================================================
#elif defined(__aarch64__)

#include <arm_neon.h>

#define SIMD_WIDTH 4
#define SIMD_SET(x) vdupq_n_f32(x)
#define SIMD_STORE(a, d) vst1q_f32(a, d)
#define SIMD_LOAD(x) vld1q_f32(x)
#define SIMD_ADD(x, y) vaddq_f32(x, y)
#define SIMD_SUB(x, y) vsubq_f32(x, y)
#define SIMD_MUL(x, y) vmulq_f32(x, y)
#define SIMD_FMA(x, y, c) vfmaq_f32(c, x, y)
#define SIMD_SQRT(x) vsqrtq_f32(x)
#define SIMD_DIV(x, y) vdivq_f32(x, y)
#define SIMD_MAX(x, y) vmaxq_f32(x, y)
#define SIMD_MIN(x, y) vminq_f32(x, y)
#define SIMD_ROUND(x) vrndnq_f32(x)

// Non-temporal store (stream store) - bypasses CPU cache, avoids RFO penalty
// For quantization output, use these to directly write to memory
#define SIMD_STREAM_STORE_U8_16(dst, val) vstnt1q_u8((uint8_t*)(dst), val)
#define SIMD_STREAM_STORE_U32_4(dst, val) vstnt1q_u32((uint32_t*)(dst), val)

// ARM NEON optimized BF16 load/store functions (SIMD-efficient)
// BF16 is essentially FP32 truncated to upper 16 bits
// Load: uint16 -> uint32 (shift left 16) -> reinterpret as float32
// Store: float32 -> round -> take upper 16 bits -> uint16

inline float32x4_t load_4_bf16_as_f32(const uint16_t* data) {
    // Load 4 uint16 values
    uint16x4_t bf16 = vld1_u16(data);
    // Extend to uint32
    uint32x4_t u32 = vmovl_u16(bf16);
    // Shift left 16 bits (BF16 → FP32 representation)
    u32 = vshlq_n_u32(u32, 16);
    // Reinterpret as float32
    return vreinterpretq_f32_u32(u32);
}

inline void store_4_f32_as_bf16_nearest(float32x4_t v, uint16_t* data) {
    // Convert to uint32 representation
    uint32x4_t u32 = vreinterpretq_u32_f32(v);

    // Round to nearest: add rounding bias then shift right 16
    // rounding_bias = ((u32 >> 16) & 1) + 0x7FFF
    uint32x4_t shifted = vshrq_n_u32(u32, 16);
    uint32x4_t lsb = vandq_u32(shifted, vdupq_n_u32(1));
    uint32x4_t bias = vaddq_u32(lsb, vdupq_n_u32(0x7FFF));
    uint32x4_t rounded = vaddq_u32(u32, bias);
    rounded = vshrq_n_u32(rounded, 16);

    // Narrow to uint16 and store
    uint16x4_t bf16 = vmovn_u32(rounded);
    vst1_u16(data, bf16);
}

union AVX_Data {
    float32x4_t data;
};

// ARM NEON simd_load/store templates
template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_store(T* dst, AVX_Data* src) {
    for (size_t i = 0; i < span; ++i) { vst1q_f32(dst + SIMD_WIDTH * i, src[i].data); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> simd_store(T* dst, AVX_Data* src) {
    for (size_t i = 0; i < span; ++i) {
        store_4_f32_as_bf16_nearest(src[i].data, reinterpret_cast<uint16_t*>(dst + SIMD_WIDTH * i));
    }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, float>, void> simd_load(AVX_Data* dst, T* src) {
    for (size_t i = 0; i < span; ++i) { dst[i].data = vld1q_f32(src + SIMD_WIDTH * i); }
}

template <int span, typename T>
inline typename std::enable_if_t<std::is_same_v<T, c10::BFloat16>, void> simd_load(AVX_Data* dst, T* src) {
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = load_4_bf16_as_f32(reinterpret_cast<const uint16_t*>(src + SIMD_WIDTH * i));
    }
}

// ARM NEON SIMD arithmetic operations (note: vfmaq_f32 is c + x*y)
template <int span> inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data* src_a) { for (size_t i = 0; i < span; ++i) { dst[i].data = vfmaq_f32(src_a[i].data, src_m_l[i].data, src_m_r.data); } }
template <int span> inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data src_a) { for (size_t i = 0; i < span; ++i) { dst[i].data = vfmaq_f32(src_a.data, src_m_l[i].data, src_m_r.data); } }
template <int span> inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vmulq_f32(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vmulq_f32(src_a_l[i].data, src_a_r[i].data); } }
template <int span> inline void simd_div(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vdivq_f32(src_a_l[i].data, src_a_r[i].data); } }
template <int span> inline void simd_sqrt(AVX_Data* dst, AVX_Data* src) { for (size_t i = 0; i < span; ++i) { dst[i].data = vsqrtq_f32(src[i].data); } }
template <int span> inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vaddq_f32(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vaddq_f32(src_a_l[i].data, src_a_r[i].data); } }
template <int span> inline void simd_sub(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vsubq_f32(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_max(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vmaxq_f32(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_min(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = vminq_f32(src_a_l[i].data, src_a_r.data); } }
template <int span> inline void simd_round(AVX_Data* dst, AVX_Data* src) { for (size_t i = 0; i < span; ++i) { dst[i].data = vrndnq_f32(src[i].data); } }

// ARM NEON quantization pack helpers
inline void pack_int8_neon(uint8_t* dst, float32x4_t vals, float delta, float z) {
    float32x4_t delta_v = vdupq_n_f32(delta);
    float32x4_t z_v = vdupq_n_f32(z);
    float32x4_t delta_inv_v = vdupq_n_f32(1.0f / delta);
    float32x4_t zero_v = vdupq_n_f32(0.0f);
    float32x4_t max_v = vdupq_n_f32(255.0f);

    float32x4_t sub = vsubq_f32(vals, z_v);
    float32x4_t scaled = vmulq_f32(sub, delta_inv_v);
    float32x4_t rounded = vrndnq_f32(scaled);
    float32x4_t clamped = vmaxq_f32(zero_v, vminq_f32(rounded, max_v));
    int32x4_t q_int = vcvtq_s32_f32(clamped);

    dst[0] = (uint8_t)vgetq_lane_s32(q_int, 0);
    dst[1] = (uint8_t)vgetq_lane_s32(q_int, 1);
    dst[2] = (uint8_t)vgetq_lane_s32(q_int, 2);
    dst[3] = (uint8_t)vgetq_lane_s32(q_int, 3);
}

inline void pack_int4_neon(uint8_t* dst, float32x4_t vals0, float32x4_t vals1, float delta, float z) {
    float32x4_t delta_v = vdupq_n_f32(delta);
    float32x4_t z_v = vdupq_n_f32(z);
    float32x4_t delta_inv_v = vdupq_n_f32(1.0f / delta);
    float32x4_t zero_v = vdupq_n_f32(0.0f);
    float32x4_t max_v = vdupq_n_f32(15.0f);

    float32x4_t sub0 = vsubq_f32(vals0, z_v);
    float32x4_t scaled0 = vmulq_f32(sub0, delta_inv_v);
    float32x4_t rounded0 = vrndnq_f32(scaled0);
    float32x4_t clamped0 = vmaxq_f32(zero_v, vminq_f32(rounded0, max_v));
    int32x4_t q_int0 = vcvtq_s32_f32(clamped0);

    float32x4_t sub1 = vsubq_f32(vals1, z_v);
    float32x4_t scaled1 = vmulq_f32(sub1, delta_inv_v);
    float32x4_t rounded1 = vrndnq_f32(scaled1);
    float32x4_t clamped1 = vmaxq_f32(zero_v, vminq_f32(rounded1, max_v));
    int32x4_t q_int1 = vcvtq_s32_f32(clamped1);

    dst[0] = ((uint8_t)vgetq_lane_s32(q_int0, 0) << 4) | (uint8_t)vgetq_lane_s32(q_int1, 0);
    dst[1] = ((uint8_t)vgetq_lane_s32(q_int0, 1) << 4) | (uint8_t)vgetq_lane_s32(q_int1, 1);
    dst[2] = ((uint8_t)vgetq_lane_s32(q_int0, 2) << 4) | (uint8_t)vgetq_lane_s32(q_int1, 2);
    dst[3] = ((uint8_t)vgetq_lane_s32(q_int0, 3) << 4) | (uint8_t)vgetq_lane_s32(q_int1, 3);
}

#endif // __aarch64__

//=============================================================================
// Scalar Fallback
//=============================================================================
#if !defined(__AVX512F__) && !defined(__AVX256__) && !defined(__aarch64__)

#define SIMD_WIDTH 1

union AVX_Data { float data; };

template <int span, typename T> inline void simd_store(T* dst, AVX_Data* src) { for (size_t i = 0; i < span; ++i) { dst[i] = (T)src[i].data; } }
template <int span, typename T> inline void simd_load(AVX_Data* dst, T* src) { for (size_t i = 0; i < span; ++i) { dst[i].data = (float)src[i]; } }
template <int span> inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data* src_a) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a[i].data + src_m_l[i].data * src_m_r.data; } }
template <int span> inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data src_a) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a.data + src_m_l[i].data * src_m_r.data; } }
template <int span> inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a_l[i].data * src_a_r.data; } }
template <int span> inline void simd_div(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a_l[i].data / src_a_r[i].data; } }
template <int span> inline void simd_sqrt(AVX_Data* dst, AVX_Data* src) { for (size_t i = 0; i < span; ++i) { dst[i].data = sqrtf(src[i].data); } }
template <int span> inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a_l[i].data + src_a_r.data; } }
template <int span> inline void simd_sub(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a_l[i].data - src_a_r.data; } }
template <int span> inline void simd_max(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a_l[i].data > src_a_r.data ? src_a_l[i].data : src_a_r.data; } }
template <int span> inline void simd_min(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r) { for (size_t i = 0; i < span; ++i) { dst[i].data = src_a_l[i].data < src_a_r.data ? src_a_l[i].data : src_a_r.data; } }
template <int span> inline void simd_round(AVX_Data* dst, AVX_Data* src) { for (size_t i = 0; i < span; ++i) { dst[i].data = roundf(src[i].data); } }

#endif