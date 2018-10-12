// FastNoiseSIMD_internal.h
//
// MIT License
//
// Copyright(c) 2017 Jordan Peck
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// The developer's email is jorzixdan.me2@gzixmail.com (for great email, take
// off every 'zix'.)
//
#ifndef _FastNoiseSIMD_internal_h_
#define _FastNoiseSIMD_internal_h_

//#include "simd_constants.inl"
//// Typedefs
//#include "internal_none.inl"
//#include "internal_neon.inl"
//#include "internal_sse2.inl"
//#include "internal_sse41.inl"
//#include "internal_avx2.inl"
//#include "internal_avx512.inl"

namespace FastNoise
{
namespace details
{

template<SIMDType _SIMDType>
struct PerturbValues
{
    typename SIMD<_SIMDType>::Float AmpV, FreqV, LacunarityV, GainV, NormaliseLengthV;
    int Octaves;
};


//template<SIMDType _SIMDType, NoiseType _NoiseType=SimplexFractal, FractalType _FractalType=FBM, PerturbType _PerturbType=None, CellularDistance _CellularDistance=Euclidean, CellularReturnType _CellularReturnType=Distance>
template<SIMDType _SIMDType>
class NoiseSIMD:public FastNoise::NoiseSIMD
{
public:
    typedef typename SIMD<_SIMDType>::Float Float;
    typedef typename SIMD<_SIMDType>::Int Int;

    NoiseSIMD(int seed=1337);

    static FastNoise::NoiseSIMD *create(int seed=1337);
    static float* GetEmptySet(size_t size);
    static size_t AlignedSize(size_t size);
    static const bool m_registered;

    void FillSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier) override;
    void FillSetMap(float* noiseSet, float* xMap, float* yMap, float* zMap, int xSize, int ySize, int zSize) override;

    void FillFractalSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier);
    void FillFractalSetMap(float* noiseSet, float* xMap, float* yMap, float* zMap, int xSize, int ySize, int zSize);

    void FillSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset);
    void FillFractalSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset);
    void FillWhiteNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier);

//    void FillCellularSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier);
//    void FillCellularSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset);
//    void FillSampledNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, int sampleScale);
//    void FillSampledNoiseSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset);
};
    

}//namespace details
}//namespace FastNoiseSIMD

//#include "FastNoiseSIMD_internal.inl"

//namespace FastNoise
//{
//namespace details
//{
//
////template instantiation
//template class NoiseSIMD<SIMDType::None>;
////template struct Constants<typename SIMD<SIMDType::None>::Float, typename SIMD<SIMDType::None>::Int, SIMDType::None>;
//
//#ifdef FN_COMPILE_NEON
//template class NoiseSIMD<SIMDType::Neon>;
////template struct Constants<typename SIMD<SIMDType::Neon>::Float, typename SIMD<SIMDType::Neon>::Int, SIMDType::Neon>;
//#endif
//
//#ifdef FN_COMPILE_SSE2
//template class NoiseSIMD<SIMDType::SSE2>;
////template struct Constants<typename SIMD<SIMDType::SSE2>::Float, typename SIMD<SIMDType::SSE2>::Int, SIMDType::SSE2>;
//#endif
//
//#ifdef FN_COMPILE_SSE41
//template class NoiseSIMD<SIMDType::SSE4_1>;
////template struct Constants<typename SIMD<SIMDType::SSE4_1>::Float, typename SIMD<SIMDType::SSE4_1>::Int, SIMDType::SSE4_1>;
//#endif
//
//#ifdef FN_COMPILE_AVX2
//template class NoiseSIMD<SIMDType::AVX2>;
////template struct Constants<typename SIMD<SIMDType::AVX2>::Float, typename SIMD<SIMDType::AVX2>::Int, SIMDType::AVX2>;
//#endif
//
//#ifdef FN_COMPILE_AVX512
//template class NoiseSIMD<SIMDType::AVX512>;
////template struct Constants<typename SIMD<SIMDType::AVX512>::Float, typename SIMD<SIMDType::AVX512>::Int, SIMDType::AVX512>;
//#endif
//
//}//namespace details
//}//namespace FastNoiseSIMD

#endif
