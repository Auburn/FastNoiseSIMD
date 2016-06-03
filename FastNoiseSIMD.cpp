#include "FastNoiseSIMD.h"

// Intrisic headers retroactively include others
//#include <immintrin.h> //AVX AVX2 FMA3
#include <smmintrin.h> //SSE4.1
//#include <emmintrin.h> //SSE2

#include <algorithm> // CPUid

// Macro redefinition warning
#ifdef _MSC_VER
#pragma warning(disable : 4005)
#endif

// Compile once for each instruction set
#ifdef COMPILE_NO_SIMD_FALLBACK
#define SIMD_LEVEL NO_SIMD_FALLBACK
#include "FastNoiseSIMD_internal.cpp"
#endif

#ifdef COMPILE_SSE2
#define SIMD_LEVEL SSE2
#include "FastNoiseSIMD_internal.cpp"
#endif

#ifdef COMPILE_SSE41
#define SIMD_LEVEL SSE41
#include "FastNoiseSIMD_internal.cpp"
#endif

// AVX2 compiled directly through FastNoiseSIMD_internal.cpp to allow arch:AVX flag

int FastNoiseSIMD::s_currentSIMDLevel = -1;

int GetFastestSIMD()
{
	// https://github.com/Mysticial/FeatureDetector

	int cpuInfo[4];

	__cpuidex(cpuInfo, 0, 0);
	int nIds = cpuInfo[0];

	if (nIds < 0x00000001)
		return NO_SIMD_FALLBACK;

	__cpuidex(cpuInfo, 0x00000001, 0);

	// SSE2
	if ((cpuInfo[3] & 1 << 26) == 0)
		return NO_SIMD_FALLBACK;

	// SSE41
	if ((cpuInfo[2] & 1 << 19) == 0)
		return SSE2;

	// AVX
	bool osAVXSuport =	(cpuInfo[2] & 1 << 27) != 0;
	bool cpuAVXSuport = (cpuInfo[2] & 1 << 28) != 0;

	if (osAVXSuport && cpuAVXSuport)
	{
		unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
		if ((xcrFeatureMask & 0x6) == 0)
			return SSE41;
	}
	else
		return SSE41;

	// AVX2 FMA3
	if (nIds < 0x00000007)
		return SSE41;

	bool cpuFMA3Support = (cpuInfo[2] & 1 << 12) != 0;

	__cpuidex(cpuInfo, 0x00000007, 0);
	
	bool cpuAVX2Support = (cpuInfo[1] & 1 << 5) != 0;

	if (cpuFMA3Support && cpuAVX2Support)
		return AVX2;
	else
		return SSE41;
}

FastNoiseSIMD* FastNoiseSIMD::NewFastNoiseSIMD(int seed)
{
	if (s_currentSIMDLevel < 0)
		s_currentSIMDLevel = GetFastestSIMD();

#ifdef COMPILE_AVX2
	if (s_currentSIMDLevel >= AVX2)
		return new FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(AVX2)(seed);
#endif

#ifdef COMPILE_SSE41
	if (s_currentSIMDLevel >= SSE41)
		return new FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(SSE41)(seed);
#endif

#ifdef COMPILE_SSE2
	if (s_currentSIMDLevel >= SSE2)
		return new FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(SSE2)(seed);
#endif

#ifdef COMPILE_NO_SIMD_FALLBACK
	return new FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(NO_SIMD_FALLBACK)(seed);
#else
	return nullptr;
#endif
}

void FastNoiseSIMD::FreeNoiseSet(float* floatArray)
{
	if (s_currentSIMDLevel > NO_SIMD_FALLBACK)
		_aligned_free(floatArray);
	else
		delete[] floatArray;
}

