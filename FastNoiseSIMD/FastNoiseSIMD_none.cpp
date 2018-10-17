#include "FastNoiseSIMD.h"

#include "simd_constants.inl"
#include "internal_none.inl"
#include "FastNoiseSIMD_internal.h"

#include "simd_init.inl"
#include "FastNoiseSIMD_internal.inl"

namespace FastNoise
{
namespace details
{

template class NoiseSIMD<SIMDType::None>;

}//namespace details
}//namespace FastNoise

