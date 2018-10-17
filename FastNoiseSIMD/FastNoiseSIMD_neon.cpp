#include "FastNoiseSIMD.h"
#include "FastNoiseSIMD_internal.h"

namespace FastNoise
{
namespace details
{

#ifdef FN_COMPILE_NEON
template class NoiseSIMD<SIMDType::Neon>;
#endif

}//namespace details
}//namespace FastNoiseSIMD

