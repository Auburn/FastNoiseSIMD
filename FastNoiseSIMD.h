#ifndef FASTNOISE_SIMD_H
#define FASTNOISE_SIMD_H

// Comment out lines to not compile for certain instruction sets
#define COMPILE_NO_SIMD_FALLBACK 
#define COMPILE_SSE2 
#define COMPILE_SSE41 

// To compile AVX2 set C++ code generation to use /arch:AVX on FastNoiseSIMD_internal.cpp
#define COMPILE_AVX2 
// Note: This does not break support for pre AVX CPUs, AVX code is only run if support is detected

/*
CPU instruction support

SSE2
Intel Pentium 4 - 2001
AMD Opteron/Athlon - 2003

SEE4.1
Intel Penryn - 2007
AMD Bulldozer - Q4 2011

AVX
Intel Sandy Bridge - Q1 2011
AMD Bulldozer - Q4 2011

AVX2
Intel Haswell - Q2 2013
AMD Carrizo - Q2 2015

FMA3
Intel Haswell - Q2 2013
AMD Piledriver - 2012
*/

class FastNoiseSIMD
{
public:
	enum NoiseType { Value, ValueFractal, Gradient, GradientFractal, Simplex, SimplexFractal, Cellular, CellularHQ, WhiteNoise };
	enum FractalType { FBM, Billow, RigidMulti };

	static FastNoiseSIMD* NewFastNoiseSIMD(int seed = 1337);
	static int GetSIMDLevel(void) { return s_currentSIMDLevel; }
	static void FreeNoiseSet(float* floatArray);

	void SetSeed(int seed) { m_seed = seed; }
	int GetSeed(void) const	{ return m_seed; }
	void SetFrequency(float frequency) { m_frequency = frequency; }

	void SetFractalOctaves(unsigned int octaves) { m_octaves = octaves; }
	void SetFractalLacunarity(float lacunarity) { m_lacunarity = lacunarity; }
	void SetFractalGain(float gain) { m_gain = gain; }
	void SetFractalType(FractalType fractalType) { m_fractalType = fractalType; }

	virtual float* GetGradientSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance = 1.0f) = 0;
	virtual float* GetGradientFractalSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance = 1.0f) = 0;

protected:
	int m_seed = 0;
	float m_frequency = 0.01f;
	NoiseType m_noiseType = Value;

	unsigned int m_octaves = 3;
	float m_lacunarity = 2.0f;
	float m_gain = 0.5f;
	FractalType m_fractalType = FBM;

	static int s_currentSIMDLevel;
};

#define NO_SIMD_FALLBACK 0
#define SSE2 1
#define SSE41 2
#define AVX2 3

#define FASTNOISE_SIMD_CLASS2(x) FastNoiseSIMD_L##x
#define FASTNOISE_SIMD_CLASS(level) FASTNOISE_SIMD_CLASS2(level)

namespace FastNoiseSIMD_internal
{
#ifdef COMPILE_NO_SIMD_FALLBACK
#define SIMD_LEVEL_H NO_SIMD_FALLBACK
#include "FastNoiseSIMD_internal.h"
#endif

#ifdef COMPILE_SSE2
#define SIMD_LEVEL_H SSE2
#include "FastNoiseSIMD_internal.h"
#endif

#ifdef COMPILE_SSE41
#define SIMD_LEVEL_H SSE41
#include "FastNoiseSIMD_internal.h"
#endif

#ifdef COMPILE_AVX2
#define SIMD_LEVEL_H AVX2
#include "FastNoiseSIMD_internal.h"
#endif
}
#endif
