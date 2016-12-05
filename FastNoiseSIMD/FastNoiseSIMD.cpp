// FastNoiseSIMD.cpp
//
// MIT License
//
// Copyright(c) 2016 Jordan Peck
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

#include "FastNoiseSIMD.h"
#include <assert.h>
#include <stdlib.h>

#ifdef FN_COMPILE_NO_SIMD_FALLBACK
#define SIMD_LEVEL_H FN_NO_SIMD_FALLBACK
#include "FastNoiseSIMD_internal.h"
#endif

#ifdef FN_COMPILE_ARMV7
#define SIMD_LEVEL_H FN_ARMV7
#include "FastNoiseSIMD_internal.h"
#endif

// CPUid


int FastNoiseSIMD::s_currentSIMDLevel = -1;


int GetFastestSIMD()
{
	return 1;
}


FastNoiseSIMD* FastNoiseSIMD::NewFastNoiseSIMD(int seed)
{
	GetSIMDLevel();

#ifdef FN_COMPILE_ARMV7
	if (s_currentSIMDLevel >= FN_ARMV7)
		return new FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(FN_ARMV7)(seed);
#endif

#ifdef FN_COMPILE_NO_SIMD_FALLBACK
	return new FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(FN_NO_SIMD_FALLBACK)(seed);
#endif
}

int FastNoiseSIMD::GetSIMDLevel()
{
	if (s_currentSIMDLevel < 0)
		s_currentSIMDLevel = GetFastestSIMD();

	return s_currentSIMDLevel;
}

void FastNoiseSIMD::FreeNoiseSet(float* floatArray)
{
#ifdef FN_ALIGNED_SETS
	GetSIMDLevel();

	if (s_currentSIMDLevel > FN_NO_SIMD_FALLBACK)
#ifdef _WIN32
		_aligned_free(floatArray);
#else
		free(floatArray);
#endif
	else
#endif
		delete[] floatArray;
}

int FastNoiseSIMD::AlignedSize(int size)
{
#ifdef FN_ALIGNED_SETS
	GetSIMDLevel();

#ifdef FN_COMPILE_ARMV7
	if (s_currentSIMDLevel >= FN_ARMV7)
		return FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(FN_ARMV7)::AlignedSize(size);
#endif
#endif
	return size;
}

float* FastNoiseSIMD::GetEmptySet(int size)
{
#ifdef FN_ALIGNED_SETS
	GetSIMDLevel();

#ifdef FN_COMPILE_ARMV7
	if (s_currentSIMDLevel >= FN_ARMV7)
		return FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(FN_ARMV7)::GetEmptySet(size);
#endif
#endif
	return new float[size];
}

FastNoiseVectorSet* FastNoiseSIMD::GetVectorSet(int xSize, int ySize, int zSize)
{
	FastNoiseVectorSet* vectorSet = new FastNoiseVectorSet();
	FillVectorSet(vectorSet, xSize, ySize, zSize);
	return vectorSet;
}

void FastNoiseSIMD::FillVectorSet(FastNoiseVectorSet* vectorSet, int xSize, int ySize, int zSize)
{
	assert(vectorSet);

	vectorSet->SetSize(xSize*ySize*zSize);
	vectorSet->sampleScale = 0;

	int index = 0;

	for (int ix = 0; ix < xSize; ix++)
	{
		for (int iy = 0; iy < ySize; iy++)
		{
			for (int iz = 0; iz < zSize; iz++)
			{
				vectorSet->xSet[index] = float(ix);
				vectorSet->ySet[index] = float(iy);
				vectorSet->zSet[index] = float(iz);
				index++;
			}
		}
	}
}

FastNoiseVectorSet* FastNoiseSIMD::GetSamplingVectorSet(int sampleScale, int xSize, int ySize, int zSize)
{
	FastNoiseVectorSet* vectorSet = new FastNoiseVectorSet();
	FillSamplingVectorSet(vectorSet, sampleScale, xSize, ySize, zSize);
	return vectorSet;
}

void FastNoiseSIMD::FillSamplingVectorSet(FastNoiseVectorSet* vectorSet, int sampleScale, int xSize, int ySize, int zSize)
{
	assert(vectorSet);

	if (sampleScale <= 0)
	{
		FillVectorSet(vectorSet, xSize, ySize, zSize);
		return;
	}

	vectorSet->sampleSizeX = xSize;
	vectorSet->sampleSizeY = ySize;
	vectorSet->sampleSizeZ = zSize;

	int sampleSize = 1 << sampleScale;
	int sampleMask = sampleSize - 1;

	int xSizeSample = xSize;
	int ySizeSample = ySize;
	int zSizeSample = zSize;

	if (xSizeSample & sampleMask)
		xSizeSample = (xSizeSample & ~sampleMask) + sampleSize;

	if (ySizeSample & sampleMask)
		ySizeSample = (ySizeSample & ~sampleMask) + sampleSize;

	if (zSizeSample & sampleMask)
		zSizeSample = (zSizeSample & ~sampleMask) + sampleSize;

	xSizeSample = (xSizeSample >> sampleScale) + 1;
	ySizeSample = (ySizeSample >> sampleScale) + 1;
	zSizeSample = (zSizeSample >> sampleScale) + 1;

	vectorSet->SetSize(xSizeSample*ySizeSample*zSizeSample);
	vectorSet->sampleScale = sampleScale;

	int index = 0;

	for (int ix = 0; ix < xSizeSample; ix++)
	{
		for (int iy = 0; iy < ySizeSample; iy++)
		{
			for (int iz = 0; iz < zSizeSample; iz++)
			{
				vectorSet->xSet[index] = float(ix*sampleSize);
				vectorSet->ySet[index] = float(iy*sampleSize);
				vectorSet->zSet[index] = float(iz*sampleSize);
				index++;
			}
		}
	}
}

float* FastNoiseSIMD::GetNoiseSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
{
	float* noiseSet = GetEmptySet(xSize, ySize, zSize);

	FillNoiseSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);

	return noiseSet;
}

void FastNoiseSIMD::FillNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
{
	switch (m_noiseType)
	{
	case Value:
		FillValueSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	case ValueFractal:
		FillValueFractalSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	case Perlin:
		FillPerlinSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	case PerlinFractal:
		FillPerlinFractalSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	case Simplex:
		FillSimplexSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	case SimplexFractal:
		FillSimplexFractalSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	case WhiteNoise:
		FillWhiteNoiseSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	case Cellular:
		FillCellularSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
		break;
	default:
		break;
	}
}

void FastNoiseSIMD::FillNoiseSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)
{
	switch (m_noiseType)
	{
	case Value:
		FillValueSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	case ValueFractal:
		FillValueFractalSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	case Perlin:
		FillPerlinSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	case PerlinFractal:
		FillPerlinFractalSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	case Simplex:
		FillSimplexSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	case SimplexFractal:
		FillSimplexFractalSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	case WhiteNoise:
		FillWhiteNoiseSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	case Cellular:
		FillCellularSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		break;
	default:
		break;
	}
}

float* FastNoiseSIMD::GetSampledNoiseSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, int sampleScale)
{
	float* noiseSet = GetEmptySet(xSize, ySize, zSize);

	FillSampledNoiseSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, sampleScale);

	return noiseSet;
}

#define GET_SET(f) \
float* FastNoiseSIMD::Get##f##Set(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)\
{\
	float* noiseSet = GetEmptySet(xSize, ySize, zSize);\
	\
	Fill##f##Set(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);\
	\
	return noiseSet;\
}

GET_SET(WhiteNoise)

GET_SET(Value)
GET_SET(ValueFractal)

GET_SET(Perlin)
GET_SET(PerlinFractal)

GET_SET(Simplex)
GET_SET(SimplexFractal)

GET_SET(Cellular)

void FastNoiseVectorSet::Free()
{
	size = -1;
	FastNoiseSIMD::FreeNoiseSet(xSet);
	xSet = nullptr;
	ySet = nullptr;
	zSet = nullptr;
}

void FastNoiseVectorSet::SetSize(int _size)
{
	Free();
	size = _size;

	int alignedSize = FastNoiseSIMD::AlignedSize(size);

	xSet = FastNoiseSIMD::GetEmptySet(alignedSize * 3);
	ySet = xSet + alignedSize;
	zSet = ySet + alignedSize;
}
