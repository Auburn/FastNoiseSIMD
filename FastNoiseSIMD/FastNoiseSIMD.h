// FastNoiseSIMD.h
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

// VERSION: 0.7.0

#ifndef FASTNOISE_SIMD_H
#define FASTNOISE_SIMD_H

#include "FastNoiseSIMD_export.h"

#include <vector>
#include <string>

namespace FastNoise
{

#if defined(__arm__) || defined(__aarch64__)
#   define FN_ARM
//#define FN_IOS
#   define FN_COMPILE_NEON
#   define FN_COMPILE_NO_SIMD_FALLBACK
#else

#ifdef _MSC_VER
#   if(_M_X64  == 100)
#       define FN_COMPILE_SSE2
#       define FN_COMPILE_SSE41
#       ifdef __AVX2__
#           define FN_COMPILE_AVX2
#       endif
#       define FN_COMPILE_NO_SIMD_FALLBACK
#   endif
#else
#   ifdef __SSE2__
#       define FN_COMPILE_SSE2
#   endif
#   ifdef __SSE4_1__
#       define FN_COMPILE_SSE41
#   endif
#   ifdef __AVX2__
#       define FN_COMPILE_AVX2
#   endif
#   ifdef __AVX512F__
#       define FN_COMPILE_AVX512
#   endif
#endif

// Using FMA instructions with AVX(51)2/NEON provides a small performance increase but can cause 
// minute variations in noise output compared to other SIMD levels due to higher calculation precision
// Intel compiler will always generate FMA instructions, use /Qfma- or -no-fma to disable
#define FN_USE_FMA
#endif

#define FN_CELLULAR_INDEX_MAX 3

// Using aligned sets of memory for float arrays allows faster storing of SIMD data
// Comment out to allow unaligned float arrays to be used as sets
#define FN_ALIGNED_SETS

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

AVX-512F
Intel Skylake-X - Q2 2017
*/
struct VectorSet;

enum class NoiseType { None, Value, ValueFractal, Perlin, PerlinFractal, Simplex, SimplexFractal, WhiteNoise, Cellular, Cubic, CubicFractal };
enum class FractalType { None, FBM, Billow, RigidMulti };
enum class PerturbType { None, Gradient, GradientFractal, Normalise, Gradient_Normalise, GradientFractal_Normalise };
enum class CellularDistance { None, Euclidean, Manhattan, Natural };
enum class CellularReturnType { None, Value, Distance, Distance2, Distance2Add, Distance2Sub, Distance2Mul, Distance2Div, NoiseLookup, Distance2Cave };

enum class SIMDType { None, Neon, SSE2, SSE4_1, AVX2, AVX512 };

enum class NoiseClass {Single, Fractal, Cellular};
enum class BuildType {Default, Map, Vector };

struct NoiseDetails
{
    NoiseDetails():
        seed(1337),
        frequency(0.01f),

        xScale(1.0f),
        yScale(1.0f),
        zScale(1.0f),

        octaves(3),
        lacunarity(2.0f),
        gain(0.5f),

        cellularNoiseLookupFrequency(0.2f),
        cellularDistanceIndex0(0),
        cellularDistanceIndex1(1),
        cellularJitter(0.45f)
     {}

    int  seed;
    float  frequency;

    float  xScale;
    float  yScale;
    float  zScale;

    int  octaves;
    float  lacunarity;
    float  gain;
    float fractalBounding;

    float  cellularNoiseLookupFrequency;
    int  cellularDistanceIndex0;
    int  cellularDistanceIndex1;
    float  cellularJitter;
};

struct PerturbDetails
{
    PerturbDetails():
        Amp(1.0f),
        Frequency(0.5f),
        Octaves(3),
        Lacunarity(2.0f),
        Gain(0.5f),
        NormaliseLength(1.0f)
    {}

    float Amp;
    float Frequency;

    int Octaves;
    float Lacunarity;
    float Gain;
    float FractalBounding;
    float NormaliseLength;
};

#if defined(_WIN32)
#define VECTORCALL __vectorcall
#else
#define VECTORCALL
#endif

class NoiseSIMD;
typedef NoiseSIMD *(*NewNoiseSimdFunc)(int);
typedef size_t(*AlignedSizeFunc)(size_t);
typedef float *(*GetEmptySetFunc)(size_t);

struct NoiseFuncs
{
    NoiseFuncs():createFunc(nullptr), alignedSizeFunc(nullptr), getEmptySetFunc(nullptr) {}

    NewNoiseSimdFunc createFunc;
    AlignedSizeFunc alignedSizeFunc;
    GetEmptySetFunc getEmptySetFunc;
};

class FASTNOISE_EXPORT NoiseSIMD
{
public:
    // Loads all available simd libraries
    static bool loadSimd(std::string directory);
    static size_t GetFastestSIMD();

    // Creates new NoiseSIMD for the highest supported instuction set of the CPU 
    static NoiseSIMD* New(int seed=1337);

    // Returns highest detected level of CPU support
    // 5: ARM NEON
    // 4: AVX-512F
    // 3: AVX2 & FMA3
    // 2: SSE4.1
    // 1: SSE2
    // 0: Fallback, no SIMD support
    static size_t GetSIMDLevel(void);

    // Sets the SIMD level for newly created NoiseSIMD objects
    // 5: ARM NEON
    // 4: AVX-512F
    // 3: AVX2 & FMA3
    // 2: SSE4.1
    // 1: SSE2
    // 0: Fallback, no SIMD support
    // -1: Auto-detect fastest supported (Default)
    // Caution: Setting this manually can cause crashes on CPUs that do not support that level
    // Caution: Changing this after creating NoiseSIMD objects has undefined behaviour
    static bool SetSIMDLevel(SIMDType type)
    {
        size_t index=(size_t)type;

        if(index>=m_noiseSimds.size())
            return false;
        if(!m_noiseSimds[index].createFunc)
            return false;

        s_currentSIMDLevel=index;
        return true;
    }

    // Free a noise set from memory
    static void FreeNoiseSet(float* noiseSet);

    // Create an empty (aligned) noise set for use with FillNoiseSet()
    static float* GetEmptySet(size_t size);

    // Create an empty (aligned) noise set for use with FillNoiseSet()
    static float* GetEmptySet(size_t xSize, size_t ySize, size_t zSize) { return GetEmptySet(xSize*ySize*zSize); }

    // Rounds the size up to the nearest aligned size for the current SIMD level
    static size_t AlignedSize(size_t size);


    virtual ~NoiseSIMD() {}

    // Returns seed used for all noise types
    int GetSeed(void) const { return m_noiseDetails.seed; }

    // Sets seed used for all noise types
    // Default: 1337
    void SetSeed(int seed) { m_noiseDetails.seed=seed; }

    // Sets frequency for all noise types
    // Default: 0.01
    void SetFrequency(float frequency) { m_noiseDetails.frequency=frequency; }

    //	// Sets noise return type of (Get/Fill)NoiseSet()
    //	// Default: Simplex
    void SetNoiseType(NoiseType noiseType) { m_noiseType = noiseType; }

        // Sets scaling factor for individual axis
        // Defaults: 1.0
    void SetAxisScales(float xScale, float yScale, float zScale) { m_noiseDetails.xScale=xScale; m_noiseDetails.yScale=yScale; m_noiseDetails.zScale=zScale; }


    // Sets octave count for all fractal noise types
    // Default: 3
    void SetFractalOctaves(int octaves) { m_noiseDetails.octaves=octaves; m_noiseDetails.fractalBounding=CalculateFractalBounding(m_noiseDetails.octaves, m_noiseDetails.gain); }

    // Sets octave lacunarity for all fractal noise types
    // Default: 2.0
    void SetFractalLacunarity(float lacunarity) { m_noiseDetails.lacunarity=lacunarity; }

    // Sets octave gain for all fractal noise types
    // Default: 0.5
    void SetFractalGain(float gain) { m_noiseDetails.gain=gain; m_noiseDetails.fractalBounding=CalculateFractalBounding(m_noiseDetails.octaves, m_noiseDetails.gain); }

    //	// Sets method for combining octaves in all fractal noise types
    //	// Default: FBM
    //	void SetFractalType(FractalType fractalType) { m_fractalType = fractalType; }


    //	// Sets return type from cellular noise calculations
    //	// Default: Distance
    //	void SetCellularReturnType(CellularReturnType cellularReturnType) { m_cellularReturnType = cellularReturnType; }
    //
    //	// Sets distance function used in cellular noise calculations
    //	// Default: Euclidean
    //	void SetCellularDistanceFunction(CellularDistanceFunction cellularDistanceFunction) { m_cellularDistanceFunction = cellularDistanceFunction; }

    //	// Sets the type of noise used if cellular return type is set the NoiseLookup
    //	// Default: Simplex
    //	void SetCellularNoiseLookupType(NoiseType cellularNoiseLookupType) { m_cellularNoiseLookupType = cellularNoiseLookupType; }

        // Sets relative frequency on the cellular noise lookup return type
        // Default: 0.2
    void SetCellularNoiseLookupFrequency(float cellularNoiseLookupFrequency) { m_noiseDetails.cellularNoiseLookupFrequency=cellularNoiseLookupFrequency; }

    // Sets the 2 distance indicies used for distance2 return types
    // Default: 0, 1
    // Note: index0 should be lower than index1
    // Both indicies must be >= 0, index1 must be < 4
    void SetCellularDistance2Indicies(int cellularDistanceIndex0, int cellularDistanceIndex1);

    // Sets the maximum distance a cellular point can move from it's grid position
    // Setting this high will make artifacts more common
    // Default: 0.45
    void SetCellularJitter(float cellularJitter) { m_noiseDetails.cellularJitter=cellularJitter; }


    //	// Enables position perturbing for all noise types
    //	// Default: None
    //	void SetPerturbType(PerturbType perturbType) { m_perturbType = perturbType; }

        // Sets the maximum distance the input position can be perturbed
        // Default: 1.0
    void SetPerturbAmp(float perturbAmp) { m_perturbDetails.Amp=perturbAmp/511.5f; }

    // Set the relative frequency for the perturb gradient
    // Default: 0.5
    void SetPerturbFrequency(float perturbFrequency) { m_perturbDetails.Frequency=perturbFrequency; }


    // Sets octave count for perturb fractal types
    // Default: 3
    void SetPerturbFractalOctaves(int perturbOctaves) { m_perturbDetails.Octaves=perturbOctaves; m_perturbDetails.FractalBounding=CalculateFractalBounding(m_perturbDetails.Octaves, m_perturbDetails.Gain); }

    // Sets octave lacunarity for perturb fractal types 
    // Default: 2.0
    void SetPerturbFractalLacunarity(float perturbLacunarity) { m_perturbDetails.Lacunarity=perturbLacunarity; }

    // Sets octave gain for perturb fractal types 
    // Default: 0.5
    void SetPerturbFractalGain(float perturbGain) { m_perturbDetails.Gain=perturbGain; m_perturbDetails.FractalBounding=CalculateFractalBounding(m_perturbDetails.Octaves, m_perturbDetails.Gain); }

    // Sets the length for vectors after perturb normalising 
    // Default: 1.0
    void SetPerturbNormaliseLength(float perturbNormaliseLength) { m_perturbDetails.NormaliseLength=perturbNormaliseLength; }


    static VectorSet* GetVectorSet(int xSize, int ySize, int zSize);
    static VectorSet* GetSamplingVectorSet(int sampleScale, int xSize, int ySize, int zSize);
    static void FillVectorSet(VectorSet* vectorSet, int xSize, int ySize, int zSize);
    static void FillSamplingVectorSet(VectorSet* vectorSet, int sampleScale, int xSize, int ySize, int zSize);

    float* GetNoiseSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier=1.0f);
    virtual void FillSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier=1.0f);
    virtual void FillSet(float* noiseSet, VectorSet* vectorSet, float xOffset=0.0f, float yOffset=0.0f, float zOffset=0.0f);

    static bool registerNoiseSimd(SIMDType type, NewNoiseSimdFunc createFunc, AlignedSizeFunc alignedSizeFunc, GetEmptySetFunc getEmptySetFunc);

protected:
    NoiseDetails m_noiseDetails;
//    int m_seed=1337;
//    float m_frequency=0.01f;
    NoiseType m_noiseType =NoiseType::SimplexFractal;

//    float m_xScale=1.0f;
//    float m_yScale=1.0f;
//    float m_zScale=1.0f;
//
//    int m_octaves=3;
//    float m_lacunarity=2.0f;
//    float m_gain=0.5f;
    FractalType m_fractalType =FractalType::FBM;
//    float m_fractalBounding;

    CellularDistance m_cellularDistance = CellularDistance::Euclidean;
    CellularReturnType m_cellularReturnType = CellularReturnType::Distance;
    NoiseType m_cellularNoiseLookupType = NoiseType::Simplex;
//    float m_cellularNoiseLookupFrequency=0.2f;
//    int m_cellularDistanceIndex0=0;
//    int m_cellularDistanceIndex1=1;
//    float m_cellularJitter=0.45f;

    PerturbType m_perturbType =PerturbType::None;
    //	float m_perturbAmp = 1.0f;
    //	float m_perturbFrequency = 0.5f;
    //
    //	int m_perturbOctaves = 3;
    //	float m_perturbLacunarity = 2.0f;
    //	float m_perturbGain = 0.5f;
    //	float m_perturbFractalBounding;
    //	float m_perturbNormaliseLength = 1.0f;
    PerturbDetails m_perturbDetails;

    static size_t s_currentSIMDLevel;
    static std::vector<NoiseFuncs> m_noiseSimds;
    static float CalculateFractalBounding(int octaves, float gain);
};

struct FASTNOISE_EXPORT VectorSet
{
public:
    size_t size=-1;
    float* xSet=nullptr;
    float* ySet=nullptr;
    float* zSet=nullptr;

    // Only used for sampled vector sets
    int sampleScale=0;
    int sampleSizeX=-1;
    int sampleSizeY=-1;
    int sampleSizeZ=-1;

    VectorSet() {}

    VectorSet(size_t _size) { SetSize(_size); }

    ~VectorSet() { Free(); }

    void Free();

    void SetSize(size_t _size);
};

}//namespace FastNoise

#endif
