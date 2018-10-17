#ifdef FN_COMPILE_SSE41
#include <smmintrin.h>
#endif

namespace FastNoise
{
namespace details
{

#ifdef FN_COMPILE_SSE41

template<>
struct SIMD<SIMDType::SSE4_1>:SIMD<SIMDType::SSE2>
{
    typedef typename SIMD<SIMDType::SSE2>::Float Float;
    typedef typename SIMD<SIMDType::SSE2>::Int Int;
    typedef typename SIMD<SIMDType::SSE2>::Mask Mask;

    static constexpr size_t const level() { return (size_t)SIMDType::SSE4_1; }

    static Float floor(Float a) { return _mm_floor_ps(a); }
    static Float blend(Float a, Float b, Mask mask) { return _mm_blendv_ps(a, b, cast(mask)); }
    static Int mul(Int a, Int b) { return _mm_mullo_epi32(a, b); }
};

#endif

}//namespace details
}//namespace FastNoise
