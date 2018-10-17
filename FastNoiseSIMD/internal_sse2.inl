
#ifdef FN_COMPILE_SSE2
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

namespace FastNoise
{
namespace details
{

#ifdef FN_COMPILE_SSE2
template<>
struct SIMD<SIMDType::SSE2>
{
    typedef __m128 Float;
    typedef __m128i Int;
    typedef Int Mask;
    static constexpr size_t const vectorSize() { return 4; }
    static constexpr size_t const alignment() { return 16; }
    static constexpr size_t const level() { return (size_t)SIMDType::SSE2; }

    static Float set(float a) { return _mm_set1_ps(a); }
    static Float zeroFloat() { return _mm_setzero_ps(); }
    static Int set(int a) { return _mm_set1_epi32(a); }
    static Int zeroInt() { return _mm_setzero_si128(); }
    static void zeroAll() {}

#ifdef FN_ALIGNED_SETS
    static void store(float *p, Float a) { _mm_store_ps(p, a); }
    static Float load(float *p) { return _mm_load_ps(p); }
#else
    static void store(float *p, Float a) { _mm_storeu_ps(p, a); }
    static Float load(float *p) { return _mm_loadu_ps(p); }
#endif

    static Float undefinedFloat() { return zeroFloat(); }
    static Int undefinedInt() { return zeroInt(); }

    static Float convert(Int a) { return _mm_cvtepi32_ps(a); }
    static Float cast(Int a) { return _mm_castsi128_ps(a); }
    static Int convert(Float a) { return _mm_cvtps_epi32(a); }
    static Int cast(Float a) { return  _mm_castps_si128(a); }

    static Float add(Float a, Float b) { return _mm_add_ps(a, b); }
    static Float sub(Float a, Float b) { return _mm_sub_ps(a, b); }
    static Float mulf(Float a, Float b) { 
        return _mm_mul_ps(a, b); 
    }
    static Float div(Float a, Float b) { return _mm_div_ps(a, b); }

    static Float mulAdd(Float a, Float b, Float c) { return add(mulf(a, b), c); }
    static Float nmulAdd(Float a, Float b, Float c) { return sub(c, mulf(a, b)); }
    static Float mulSub(Float a, Float b, Float c) { return sub(mulf(a, b), c); }

    static Float min(Float a, Float b) { return _mm_min_ps(a, b); }
    static Float max(Float  a, Float b) { return _mm_max_ps(a, b); }
    static Float invSqrt(Float a) { 
        return _mm_rsqrt_ps(a); 
    }

    static Mask equal(Float a, Float b) { return cast(_mm_cmpeq_ps(a, b)); }
    static Mask lessThan(Float a, Float b) { return cast(_mm_cmplt_ps(a, b)); }
    static Mask greaterThan(Float a, Float b) { return cast(_mm_cmpgt_ps(a, b)); }
    static Mask lessEqual(Float a, Float b) { return cast(_mm_cmple_ps(a, b)); }
    static Mask greaterEqual(Float a, Float b) { return cast(_mm_cmpge_ps(a, b)); }

    static Float _and(Float a, Float b) { return _mm_and_ps(a, b); }
    static Float andNot(Float a, Float b) { return _mm_andnot_ps(a, b); }
    static Float _xor(Float a, Float b) { return _mm_xor_ps(a, b); }

    static Float VECTORCALL v_floor(Float a)
    {
        Float fval=_mm_cvtepi32_ps(_mm_cvttps_epi32(a));

        return _mm_sub_ps(fval, _mm_and_ps(_mm_cmplt_ps(a, fval), Constants<Float, Int, SIMDType::SSE2>::numf_1));
    }
    static Float floor(Float a) { return v_floor(a); }
    static Float abs(Float a) { return (_and(a, cast(Constants<Float, Int, SIMDType::SSE2>::numi_0x7fffffff))); }
    static Float blend(Float a, Float b, Mask mask) { return _mm_or_ps(_mm_andnot_ps(cast(mask), a), _mm_and_ps(cast(mask), b)); }

    static Int add(Int a, Int b)
    {
        return _mm_add_epi32(a, b);
    }
    static Int sub(Int a, Int b) { return _mm_sub_epi32(a, b); }

    static Int VECTORCALL v_mul(Int a, Int b)
    {
        Float tmp1=_mm_castsi128_ps(_mm_mul_epu32(a, b)); /* mul 2,0*/
        Float tmp2=_mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4))); /* mul 3,1 */
        return _mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(2, 0, 2, 0))), _MM_SHUFFLE(3, 1, 2, 0));
    }
    static Int mul(Int a, Int b) { return v_mul(a, b); }

    static Int _and(Int a, Int b) { return _mm_and_si128(a, b); }
    static Int andNot(Int a, Int b) { return _mm_andnot_si128(a, b); }
    static Int _or(Int a, Int b) { return _mm_or_si128(a, b); }
    static Int _xor(Int a, Int b) { return _mm_xor_si128(a, b); }
    static Int _not(Int a) { return _xor(a, Constants<Float, Int, SIMDType::SSE2>::numi_0xffffffff); }

    static Int shiftR(Int a, int b) { return _mm_srai_epi32(a, b); }
    static Int shiftL(Int a, int b) { return _mm_slli_epi32(a, b); }

    static Mask equal(Int a, Int b) { return _mm_cmpeq_epi32(a, b); }
    static Mask greaterThan(Int a, Int b) { return _mm_cmpgt_epi32(a, b); }
    static Mask lessThan(Int a, Int b) { return _mm_cmpgt_epi32(b, a); }

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

#endif

}//namespace details
}//namespace FastNoise
