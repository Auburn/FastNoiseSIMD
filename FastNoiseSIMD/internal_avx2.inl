#include <immintrin.h>

namespace FastNoise
{
namespace details
{

template<>
struct SIMD<SIMDType::AVX2>
{
    typedef __m256 Float;
    typedef __m256i Int;
    typedef Int Mask;
    static constexpr size_t const vectorSize() { return 8; }
    static constexpr size_t const alignment() { return 32; }
    static constexpr size_t const level() { return (size_t)SIMDType::AVX2; }

    static Float set(float a) { return _mm256_set1_ps(a); }
    static Float zeroFloat() { return _mm256_setzero_ps(); }
    static Int set(int a) { return  _mm256_set1_epi32(a); }
    static Int zeroInt() { return _mm256_setzero_si256(); }
    static void zeroAll() {}

#ifdef FN_ALIGNED_SETS
    static void store(float *p, Float a) { _mm256_store_ps(p, a); }
    static Float load(float *p) { return _mm256_load_ps(p); }
#else
    static void store(float *p, Float a) { _mm256_storeu_ps(p, a); }
    static Float load(float *p) { return _mm256_loadu_ps(p); }
#endif

    static Float undefinedFloat() { return _mm256_undefined_ps(); }
    static Int undefinedInt() { return _mm256_undefined_si256(); }

    static Float convert(Int a) { return _mm256_cvtepi32_ps(a); }
    static Float cast(Int a) { return _mm256_castsi256_ps(a); }
    static Int convert(Float a) { return _mm256_cvtps_epi32(a); }
    static Int cast(Float a) { return  _mm256_castps_si256(a); }

    static Float add(Float a, Float b) { return _mm256_add_ps(a, b); }
    static Float sub(Float a, Float b) { return _mm256_sub_ps(a, b); }
    static Float mulf(Float a, Float b) { return _mm256_mul_ps(a, b); }
    static Float div(Float a, Float b) { return _mm256_div_ps(a, b); }

    static Float mulAdd(Float a, Float b, Float c) { return _mm256_fmadd_ps(a, b, c); }
    static Float nmulAdd(Float a, Float b, Float c) { return _mm256_fnmadd_ps(a, b, c); }
    static Float mulSub(Float a, Float b, Float c) { return _mm256_fmsub_ps(a, b, c); }

    static Float min(Float a, Float b) { return _mm256_min_ps(a, b); }
    static Float max(Float  a, Float b) { return _mm256_max_ps(a, b); }
    static Float invSqrt(Float a) { return _mm256_rsqrt_ps(a); }

    static Mask equal(Float a, Float b) { return cast(_mm256_cmp_ps(a, b, _CMP_EQ_OQ)); }
    static Mask lessThan(Float a, Float b) { return cast(_mm256_cmp_ps(a, b, _CMP_LT_OQ)); }
    static Mask greaterThan(Float a, Float b) { return cast(_mm256_cmp_ps(a, b, _CMP_GT_OQ)); }
    static Mask lessEqual(Float a, Float b) { return cast(_mm256_cmp_ps(a, b, _CMP_LE_OQ)); }
    static Mask greaterEqual(Float a, Float b) { return cast(_mm256_cmp_ps(a, b, _CMP_GE_OQ)); }

    static Float _and(Float a, Float b) { return _mm256_and_ps(a, b); }
    static Float andNot(Float a, Float b) { return _mm256_andnot_ps(a, b); }
    static Float _xor(Float a, Float b) { return _mm256_xor_ps(a, b); }

    static Float floor(Float a) { return _mm256_floor_ps(a); }
    static Float abs(Float a) { return (_and(a, cast(Constants<Float, Int, SIMDType::AVX2>::numi_0x7fffffff))); }
    static Float blend(Float a, Float b, Mask mask) { return _mm256_blendv_ps(a, b, cast(mask)); }

    static Int add(Int a, Int b) { return _mm256_add_epi32(a, b); }
    static Int sub(Int a, Int b) { return _mm256_sub_epi32(a, b); }
    static Int mul(Int a, Int b) { return _mm256_mullo_epi32(a, b); }

    static Int _and(Int a, Int b) { return _mm256_and_si256(a, b); }
    static Int andNot(Int a, Int b) { return _mm256_andnot_si256(a, b); }
    static Int _or(Int a, Int b) { return _mm256_or_si256(a, b); }
    static Int _xor(Int a, Int b) { return _mm256_xor_si256(a, b); }
    static Int _not(Int a) { return _xor(a, Constants<Float, Int, SIMDType::AVX2>::numi_0xffffffff); }

    static Int shiftR(Int a, int b) { return _mm256_srai_epi32(a, b); }
    static Int shiftL(Int a, int b) { return _mm256_slli_epi32(a, b); }

    static Mask equal(Int a, Int b) { return _mm256_cmpeq_epi32(a, b); }
    static Mask greaterThan(Int a, Int b) { return _mm256_cmpgt_epi32(a, b); }
    static Mask lessThan(Int a, Int b) { return _mm256_cmpgt_epi32(b, a); }

    static Mask maskAnd(Mask a, Mask b) { return _and(a,b); }
    static Mask maskAndNot(Mask a, Mask b) { return andNot(a, b); }
    static Mask maskOr(Mask a, Mask b) { return _or(a,b); }
    static Mask maskNot(Mask a) { return _not(a); }

    static Float mask(Mask m, Float a) { return _and(cast(m), a); }
    static Float maskAdd(Mask m, Float a, Float b) { return add(a, _and(cast(m), b)); }
    static Float maskSub(Mask m, Float a, Float b) { return sub(a, _and(cast(m), b)); }

    static Int maskAdd(Mask m, Int a, Int b) { return add(a, _and(m, b)); }
    static Int maskSub(Mask m, Int a, Int b) { return sub(a, _and(m, b)); }
};

}//namespace details
}//namespace FastNoise
