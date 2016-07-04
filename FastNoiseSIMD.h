// FastNoiseSIMD.h
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

#ifndef FASTNOISE_SIMD_H
#define FASTNOISE_SIMD_H

// Comment out lines to not compile for certain instruction sets
#define FN_COMPILE_NO_SIMD_FALLBACK
#define FN_COMPILE_SSE2
#define FN_COMPILE_SSE41

// To compile AVX2 set C++ code generation to use /arch:AVX(2) on FastNoiseSIMD_avx2.cpp
#define FN_COMPILE_AVX2
// Note: This does not break support for pre AVX CPUs, AVX code is only run if support is detected

// Using aligned sets of memory for float arrays allows faster storing of SIMD data
// Comment out to allow unaligned float arrays to be used as sets
#define FN_ALIGNED_SETS

// Using FMA3 instructions with AVX2 provides a small performance increase but can cause 
// tiny variations in noise output compared to other SIMD levels due to higher calculation precision
#define FN_USE_FMA3

// Reduced minimum of zSize from 8 to 4 when not using a vector set
// Causes slightly performance loss on non-"mulitple of 8" zSize
//#define FN_MIN_Z_4

/*
Tested Compilers:
-MSVC v120/v140
-Intel 16.0
-GCC 4.7 Linux
-Clang MacOSX

CPU instruction support:

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

struct FastNoiseVectorSet;

class FastNoiseSIMD
{
public:
	enum NoiseType { Value, ValueFractal, Gradient, GradientFractal, Simplex, SimplexFractal, WhiteNoise, Cellular };
	enum FractalType { FBM, Billow, RigidMulti };

	enum CellularDistanceFunction { Euclidean, Manhattan, Natural };
	enum CellularReturnType { CellValue, Distance, Distance2, Distance2Add, Distance2Sub, Distance2Mul, Distance2Div };

	// Creates new FastNoiseSIMD for the highest supported instuction set of the CPU 
	static FastNoiseSIMD* NewFastNoiseSIMD(int seed = 1337);

	// Returns highest detected level of CPU support
	// 3: AVX2 & FMA3
	// 2: SSE4.1
	// 1: SSE2
	// 0: Fallback, no SIMD support
	static int GetSIMDLevel(void);

	// Sets the SIMD level for newly created FastNoiseSIMD objects
	// 3: AVX2 & FMA3
	// 2: SSE4.1
	// 1: SSE2
	// 0: Fallback, no SIMD support
	// -1: Auto-detect fastest supported (Default)
	// Caution: Setting this manually can cause crashes on CPUs that do not support that level
	// Caution: Changing this after creating FastNoiseSIMD objects has undefined behaviour
	static void SetSIMDLevel(int level) { s_currentSIMDLevel = level; }

	// Free a noise set from memory
	static void FreeNoiseSet(float* noiseSet);

	// Create an empty (aligned) noise set for use with FillNoiseSet()
	static float* GetEmptySet(int size);

	// Create an empty (aligned) noise set for use with FillNoiseSet()
	static float* GetEmptySet(int xSize, int ySize, int zSize) { return GetEmptySet(xSize*ySize*zSize); }

	static int AlignedSize(int size);


	// Returns seed used for all noise types
	int GetSeed(void) const { return m_seed; }

	// Sets seed used for all noise types
	// Default: 1337
	void SetSeed(int seed) { m_seed = seed; }

	// Sets frequency for all noise types
	// Default: 0.01
	void SetFrequency(float frequency) { m_frequency = frequency; }

	// Sets noise return type of (Get/Fill)NoiseSet()
	// Default: Simplex
	void SetNoiseType(NoiseType noiseType) { m_noiseType = noiseType; }

	// Sets scaling factor for individual axis
	// Defaults: 1.0
	void SetAxisScales(float xScale, float yScale, float zScale) { m_xScale = xScale; m_yScale = yScale; m_zScale = zScale; }


	// Sets octave count for all fractal noise types
	// Default: 3
	void SetFractalOctaves(unsigned int octaves) { m_octaves = octaves; }

	// Sets octave lacunarity for all fractal noise types
	// Default: 2.0
	void SetFractalLacunarity(float lacunarity) { m_lacunarity = lacunarity; }

	// Sets octave gain for all fractal noise types
	// Default: 0.5
	void SetFractalGain(float gain) { m_gain = gain; }

	// Sets method for combining octaves in all fractal noise types
	// Default: FBM
	void SetFractalType(FractalType fractalType) { m_fractalType = fractalType; }


	// Sets return type from cellular noise calculations
	// Default: Distance
	void SetCellularReturnType(CellularReturnType cellularReturnType) { m_cellularReturnType = cellularReturnType; }

	// Sets distance function used in cellular noise calculations
	// Default: Euclidean
	void SetCellularDistanceFunction(CellularDistanceFunction cellularDistanceFunction) { m_cellularDistanceFunction = cellularDistanceFunction; }

	static FastNoiseVectorSet* GetVectorSet(int xSize, int ySize, int zSize);
	static FastNoiseVectorSet* GetSamplingVectorSet(int sampleScale, int xSize, int ySize, int zSize);
	static void FillVectorSet(FastNoiseVectorSet* vectorSet, int xSize, int ySize, int zSize);
	static void FillSamplingVectorSet(FastNoiseVectorSet* vectorSet, int sampleScale, int xSize, int ySize, int zSize);

	float* GetNoiseSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	void FillNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	void FillNoiseSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f);

	float* GetSampledNoiseSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, int sampleScale);
	virtual void FillSampledNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, int sampleScale) = 0;
	virtual void FillSampledNoiseSet(float* noiseSet, FastNoiseVectorSet* vectorSet, int xSize, int ySize, int zSize, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;

	float* GetWhiteNoiseSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	virtual void FillWhiteNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillWhiteNoiseSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;

	float* GetValueSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	float* GetValueFractalSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	virtual void FillValueSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillValueFractalSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillValueSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;
	virtual void FillValueFractalSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;

	float* GetGradientSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	float* GetGradientFractalSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	virtual void FillGradientSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillGradientFractalSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillGradientSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;
	virtual void FillGradientFractalSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;

	float* GetSimplexSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	float* GetSimplexFractalSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	virtual void FillSimplexSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillSimplexFractalSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillSimplexSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;
	virtual void FillSimplexFractalSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;

	float* GetCellularSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f);
	virtual void FillCellularSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier = 1.0f) = 0;
	virtual void FillCellularSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset = 0.0f, float yOffset = 0.0f, float zOffset = 0.0f) = 0;

protected:
	int m_seed = 1337;
	float m_frequency = 0.01f;
	NoiseType m_noiseType = SimplexFractal;

	float m_xScale = 1.0f;
	float m_yScale = 1.0f;
	float m_zScale = 1.0f;

	unsigned int m_octaves = 3;
	float m_lacunarity = 2.0f;
	float m_gain = 0.5f;
	FractalType m_fractalType = FBM;

	CellularDistanceFunction m_cellularDistanceFunction = Euclidean;
	CellularReturnType m_cellularReturnType = Distance;

	static int s_currentSIMDLevel;
};

struct FastNoiseVectorSet
{
	int size = -1;
	int sampleScale = 0;
	float* xSet = nullptr;
	float* ySet = nullptr;
	float* zSet = nullptr;

	FastNoiseVectorSet() {}

	FastNoiseVectorSet(int _size) { SetSize(_size); }

	~FastNoiseVectorSet() { Free(); }

	void Free();

	void SetSize(int _size);
};

#define FN_NO_SIMD_FALLBACK 0
#define FN_SSE2 1
#define FN_SSE41 2
#define FN_AVX2 3
#endif
