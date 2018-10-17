#include "FastNoiseSIMD.h"

#include "simd_constants.inl"
#include "internal_none.inl"
#include "internal_avx2.inl"
#include "FastNoiseSIMD_internal.h"

#include "simd_init.inl"
#include "FastNoiseSIMD_internal.inl"

namespace FastNoise
{
namespace details
{

#ifdef FN_COMPILE_AVX2
template class NoiseSIMD<SIMDType::AVX2>;
//template struct Constants<typename SIMD<SIMDType::AVX2>::Float, typename SIMD<SIMDType::AVX2>::Int, SIMDType::AVX2>;
#endif

}//namespace details
}//namespace FastNoiseSIMD
