#include "FastNoiseSIMD.h"

#include "simd_constants.inl"
#include "internal_none.inl"
#include "internal_sse2.inl"
#include "internal_sse41.inl"
#include "FastNoiseSIMD_internal.h"

#include "simd_init.inl"
#include "FastNoiseSIMD_internal.inl"

namespace FastNoise
{
namespace details
{

#ifdef FN_COMPILE_SSE41
template class NoiseSIMD<SIMDType::SSE4_1>;
//template struct Constants<typename SIMD<SIMDType::SSE4_1>::Float, typename SIMD<SIMDType::SSE4_1>::Int, SIMDType::SSE4_1>;
#endif

}//namespace details
}//namespace FastNoiseSIMD
