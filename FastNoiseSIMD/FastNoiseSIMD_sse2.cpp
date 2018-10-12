#include "FastNoiseSIMD.h"

#include "simd_constants.inl"
#include "internal_none.inl"
#include "internal_sse2.inl"
#include "FastNoiseSIMD_internal.h"

#include "simd_init.inl"
#include "FastNoiseSIMD_internal.inl"

namespace FastNoise
{
namespace details
{

#ifdef FN_COMPILE_SSE2

template class NoiseSIMD<SIMDType::SSE2>;

#endif

}//namespace details
}//namespace FastNoiseSIMD
