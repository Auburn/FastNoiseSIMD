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
    void FillWhiteNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier);

    void FillSet(float* noiseSet, VectorSet* vectorSet, float xOffset, float yOffset, float zOffset);
//    void FillSampledNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, int sampleScale);
//    void FillSampledNoiseSet(float* noiseSet, VectorSet* vectorSet, float xOffset, float yOffset, float zOffset);
};
    

}//namespace details
}//namespace FastNoiseSIMD

#endif
