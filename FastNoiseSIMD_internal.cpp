// FastNoiseSIMD_internal.cpp
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

#ifndef SIMD_LEVEL
#include "FastNoiseSIMD.h"
#endif

#if defined(SIMD_LEVEL) || defined(FN_COMPILE_AVX2)

#ifndef SIMD_LEVEL
#define SIMD_LEVEL FN_AVX2
#include <immintrin.h>
#ifndef __AVX__
#ifdef __GNUC__
#error To compile AVX2 set custom build commands "$compiler $options $includes -c $file -o $object -march=core-avx2" on FastNoiseSIMD_internal.cpp, or remove "#define FN_COMPILE_AVX2" from FastNoiseSIMD.h
#else
#error To compile AVX2 set C++ code generation to use /arch:AVX(2) on FastNoiseSIMD_internal.cpp, or remove "#define FN_COMPILE_AVX2" from FastNoiseSIMD.h
#endif
#endif

// GCC build settings
#elif defined(__GNUC__) && SIMD_LEVEL != FN_NO_SIMD_FALLBACK
#pragma GCC push_options

#if SIMD_LEVEL == FN_SSE41
#pragma GCC target ("sse4.1")
#elif SIMD_LEVEL == FN_SSE2
#pragma GCC target ("sse2")
#endif
#endif

// Per SIMD level var/function naming
#define L_TYPE2(t, l) t##_L##l
#define L_TYPE(t, l) L_TYPE2(t, l)
#define SIMDf L_TYPE(SIMDf, SIMD_LEVEL)
#define SIMDi L_TYPE(SIMDi, SIMD_LEVEL)

#define L_uSIMD2(s) u##s
#define L_uSIMD(s) L_uSIMD2(s)
#define uSIMDf L_uSIMD(SIMDf)
#define uSIMDi L_uSIMD(SIMDi)

#define L_SIMD_NUM2(n, s) s##_##n
#define L_SIMD_NUM(n, s) L_SIMD_NUM2(n, s)
#define SIMDf_NUM(n) L_SIMD_NUM(n, SIMDf)
#define SIMDi_NUM(n) L_SIMD_NUM(n, SIMDi)

#define L_VAR2(x, l) L##l##_##x
#define L_VAR(x, l) L_VAR2(x, l)
#define VAR(x) L_VAR(x, SIMD_LEVEL)
#define FUNC(x) VAR(FUNC_##x)

#define SIMD_LEVEL_CLASS FastNoiseSIMD_internal::FASTNOISE_SIMD_CLASS(SIMD_LEVEL)

// Typedefs
#if SIMD_LEVEL == FN_AVX2
#define VECTOR_SIZE 8
#define MEMORY_ALIGNMENT 32
typedef __m256 SIMDf;
typedef __m256i SIMDi;
#define SIMDf_SET(a) _mm256_set1_ps(a)
#define SIMDf_SET_ZERO() _mm256_setzero_ps()
#define SIMDi_SET(a) _mm256_set1_epi32(a)
#define SIMDi_SET_ZERO() _mm256_setzero_si256()

#elif SIMD_LEVEL >= FN_SSE2
#define VECTOR_SIZE 4
#define MEMORY_ALIGNMENT 16
typedef __m128 SIMDf;
typedef __m128i SIMDi;
#define SIMDf_SET(a) _mm_set1_ps(a)
#define SIMDf_SET_ZERO() _mm_setzero_ps()
#define SIMDi_SET(a) _mm_set1_epi32(a)
#define SIMDi_SET_ZERO() _mm_setzero_si128()

#else // Fallback to float/int
#define VECTOR_SIZE 1
#define MEMORY_ALIGNMENT 4
typedef float SIMDf;
typedef int SIMDi;
#define SIMDf_SET(a) (a)
#define SIMDf_SET_ZERO() (0)
#define SIMDi_SET(a) (a)
#define SIMDi_SET_ZERO() (0)
#endif

// Memory Allocation
#if SIMD_LEVEL > FN_NO_SIMD_FALLBACK && defined(FN_ALIGNED_SETS)
#ifdef _WIN32
#define SIMD_ALIGNED_SET(floatP, floatCount) floatP = (float*)_aligned_malloc((floatCount)* sizeof(float), MEMORY_ALIGNMENT)
#else
#define SIMD_ALIGNED_SET(floatP, floatCount) posix_memalign((void**)&floatP, MEMORY_ALIGNMENT, (floatCount)* sizeof(float))
#endif
#else
#define SIMD_ALIGNED_SET(floatP, floatCount) floatP = new float[floatCount]
#endif

union uSIMDf
{
	SIMDf m;
	float a[VECTOR_SIZE];
};

union uSIMDi
{
	SIMDi m;
	int a[VECTOR_SIZE];
};

static SIMDi SIMDi_NUM(0);
static SIMDf SIMDf_NUM(1);
static SIMDi SIMDi_NUM(0xffffffff);

// SIMD functions
#if SIMD_LEVEL >= FN_AVX2

#ifdef FN_ALIGNED_SETS
#define SIMDf_STORE(p,a) _mm256_store_ps(p,a)
#else
#define SIMDf_STORE(p,a) _mm256_storeu_ps(p,a)
#endif
#define SIMDf_LOAD(p) _mm256_load_ps(p)

#define SIMDf_ADD(a,b) _mm256_add_ps(a,b)
#define SIMDf_SUB(a,b) _mm256_sub_ps(a,b)
#define SIMDf_MUL(a,b) _mm256_mul_ps(a,b)
#define SIMDf_DIV(a,b) _mm256_div_ps(a,b)

#define SIMDf_LESS_THAN(a,b) _mm256_cmp_ps(a,b,_CMP_LT_OQ)
#define SIMDf_GREATER_THAN(a,b) _mm256_cmp_ps(a,b,_CMP_GT_OQ)
#define SIMDf_LESS_EQUAL(a,b) _mm256_cmp_ps(a,b,_CMP_LE_OQ)
#define SIMDf_GREATER_EQUAL(a,b) _mm256_cmp_ps(a,b,_CMP_GE_OQ)

#define SIMDf_FLOOR(a) _mm256_floor_ps(a)
#define SIMDf_BLENDV(a,b,mask) _mm256_blendv_ps(a,b,mask)
#define SIMDf_GATHER(p,a) _mm256_i32gather_ps(p,a,4)

#define SIMDi_ADD(a,b) _mm256_add_epi32(a,b)
#define SIMDi_SUB(a,b) _mm256_sub_epi32(a,b)
#define SIMDi_MUL(a,b) _mm256_mullo_epi32(a,b)

#define SIMDi_AND(a,b) _mm256_and_si256(a,b)
#define SIMDi_AND_NOT(a,b) _mm256_andnot_si256(a,b)
#define SIMDi_OR(a,b) _mm256_or_si256(a,b)
#define SIMDi_XOR(a,b) _mm256_xor_si256(a,b)
#define SIMDi_NOT(a) SIMDi_XOR(a,SIMDi_NUM(0xffffffff))

#define SIMDi_SHIFT_R(a, b) _mm256_srli_epi32(a, b)
#define SIMDi_SHIFT_L(a, b) _mm256_slli_epi32(a, b)

#define SIMDi_EQUAL(a,b) _mm256_cmpeq_epi32(a,b)
#define SIMDi_GREATER_THAN(a,b) _mm256_cmpgt_epi32(a,b)
#define SIMDi_LESS_THAN(a,b) _mm256_cmpgt_epi32(b,a)

#define SIMDf_CONVERT_TO_FLOAT(a) _mm256_cvtepi32_ps(a)
#define SIMDf_CAST_TO_FLOAT(a) _mm256_castsi256_ps(a)
#define SIMDi_CONVERT_TO_INT(a) _mm256_cvtps_epi32(a)
#define SIMDi_CAST_TO_INT(a) _mm256_castps_si256(a)

#elif SIMD_LEVEL >= FN_SSE2

#ifdef FN_ALIGNED_SETS
#define SIMDf_STORE(p,a) _mm_store_ps(p,a)
#else
#define SIMDf_STORE(p,a) _mm_storeu_ps(p,a)
#endif
#define SIMDf_LOAD(p) _mm_load_ps(p)

#define SIMDf_ADD(a,b) _mm_add_ps(a,b)
#define SIMDf_SUB(a,b) _mm_sub_ps(a,b)
#define SIMDf_MUL(a,b) _mm_mul_ps(a,b)
#define SIMDf_DIV(a,b) _mm_div_ps(a,b)

#define SIMDf_LESS_THAN(a,b) _mm_cmplt_ps(a,b)
#define SIMDf_GREATER_THAN(a,b) _mm_cmpgt_ps(a,b)
#define SIMDf_LESS_EQUAL(a,b) _mm_cmple_ps(a,b)
#define SIMDf_GREATER_EQUAL(a,b) _mm_cmpge_ps(a,b)

#if SIMD_LEVEL == FN_SSE41
#define SIMDi_MUL(a,b) _mm_mullo_epi32(a,b)
#define SIMDf_FLOOR(a) _mm_floor_ps(a)
#define SIMDf_BLENDV(a,b,mask) _mm_blendv_ps(a,b,mask)
#else
static SIMDi FUNC(MUL)(const SIMDi& a, const SIMDi& b)
{
	__m128 tmp1 = _mm_castsi128_ps(_mm_mul_epu32(a, b)); /* mul 2,0*/
	__m128 tmp2 = _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4))); /* mul 3,1 */
	return _mm_castps_si128(_mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(2, 0, 2, 0)));
}
#define SIMDi_MUL(a,b) FUNC(MUL)(a,b)

static SIMDf FUNC(FLOOR)(const SIMDf& a)
{
	__m128 fval = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));

	return _mm_sub_ps(fval, _mm_and_ps(_mm_cmplt_ps(a, fval), SIMDf_NUM(1)));
}
#define SIMDf_FLOOR(a) FUNC(FLOOR)(a)

static SIMDf FUNC(BLENDV)(const SIMDf& a, const SIMDf& b, const SIMDf& mask)
{
	SIMDf maskInv = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(mask), SIMDi_NUM(0)));

	return _mm_or_ps(_mm_and_ps(maskInv, a), _mm_andnot_ps(maskInv, b));
}

#define SIMDf_BLENDV(a,b,mask) FUNC(BLENDV)(a,b,mask)
#endif

static SIMDf FUNC(GATHER)(const float* p, const SIMDi& a)
{
	const uSIMDi* m = reinterpret_cast<const uSIMDi*>(&a);
	uSIMDf r;

	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		r.a[i] = *(p + m->a[i]);
	}
	return r.m;
}
#define SIMDf_GATHER(p,a) FUNC(GATHER)(p, a)

#define SIMDi_ADD(a,b) _mm_add_epi32(a,b)
#define SIMDi_SUB(a,b) _mm_sub_epi32(a,b)

#define SIMDi_AND(a,b) _mm_and_si128(a,b)
#define SIMDi_AND_NOT(a,b) _mm_andnot_si128(a,b)
#define SIMDi_OR(a,b) _mm_or_si128(a,b)
#define SIMDi_XOR(a,b) _mm_xor_si128(a,b)
#define SIMDi_NOT(a) SIMDi_XOR(a,SIMDi_NUM(0xffffffff))

#define SIMDi_SHIFT_R(a,b) _mm_srli_epi32(a, b)
#define SIMDi_SHIFT_L(a,b) _mm_slli_epi32(a, b)

#define SIMDi_EQUAL(a,b) _mm_cmpeq_epi32(a,b)
#define SIMDi_GREATER_THAN(a,b) _mm_cmpgt_epi32(a,b)
#define SIMDi_LESS_THAN(a,b) _mm_cmpgt_epi32(b,a)

#define SIMDf_CONVERT_TO_FLOAT(a) _mm_cvtepi32_ps(a)
#define SIMDf_CAST_TO_FLOAT(a) _mm_castsi128_ps(a)
#define SIMDi_CONVERT_TO_INT(a) _mm_cvtps_epi32(a)
#define SIMDi_CAST_TO_INT(a) _mm_castps_si128(a)

#else // Fallback

inline static int FUNC(CAST_TO_INT)(float f) { return *reinterpret_cast<int*>(&f); }
inline static float FUNC(CAST_TO_FLOAT)(int i) { return *reinterpret_cast<float*>(&i); }
#define SIMDi_CAST_TO_INT(a) FUNC(CAST_TO_INT)(a)
#define SIMDf_CAST_TO_FLOAT(a) FUNC(CAST_TO_FLOAT)(a)

#define SIMDf_STORE(p,a) (*(p) = a)
#define SIMDf_LOAD(p) (*p)

#define SIMDf_ADD(a,b) ((a) + (b))
#define SIMDf_SUB(a,b) ((a) - (b))
#define SIMDf_MUL(a,b) ((a) * (b))
#define SIMDf_DIV(a,b) ((a) / (b))

#define SIMDf_LESS_THAN(a,b) (((a) < (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)
#define SIMDf_GREATER_THAN(a,b) (((a) > (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)
#define SIMDf_LESS_EQUAL(a,b) (((a) <= (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)
#define SIMDf_GREATER_EQUAL(a,b) (((a) >= (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)

#define SIMDf_FLOOR(a) floor(a)
#define SIMDf_BLENDV(a,b,mask) (SIMDi_CAST_TO_INT(mask) ? (b) : (a))
#define SIMDf_GATHER(p,a) (*(reinterpret_cast<const float*>(p)+(a)))

#define SIMDi_ADD(a,b) ((a) + (b))
#define SIMDi_SUB(a,b) ((a) - (b))
#define SIMDi_MUL(a,b) ((a) * (b))

#define SIMDi_AND(a,b) ((a) & (b))
#define SIMDi_AND_NOT(a,b) (~(a) & (b))
#define SIMDi_OR(a,b) ((a) | (b))
#define SIMDi_XOR(a,b) ((a) ^ (b))
#define SIMDi_NOT(a) (~(a))

#define SIMDi_SHIFT_R(a, b) ((a) >> (b))
#define SIMDi_SHIFT_L(a, b) ((a) << (b))

#define SIMDi_EQUAL(a,b) (((a) == (b)) ? 0xFFFFFFFF : 0)
#define SIMDi_GREATER_THAN(a,b) (((a) > (b)) ? 0xFFFFFFFF : 0)
#define SIMDi_LESS_THAN(a,b) (((a) < (b)) ? 0xFFFFFFFF : 0)

#define SIMDi_CONVERT_TO_INT(a) static_cast<int>(a)
#define SIMDf_CONVERT_TO_FLOAT(a) static_cast<float>(a)
#endif

#define SIMDf_AND(a,b) SIMDf_CAST_TO_FLOAT(SIMDi_AND(SIMDi_CAST_TO_INT(a),SIMDi_CAST_TO_INT(b)))
#define SIMDf_ABS(a) SIMDf_CAST_TO_FLOAT(SIMDi_AND(SIMDi_CAST_TO_INT(a),SIMDi_NUM(0x7fffffff)))

// FMA2
#if SIMD_LEVEL == FN_AVX2
#define SIMDf_MUL_ADD(a,b,c) _mm256_fmadd_ps(a,b,c)
#define SIMDf_MUL_SUB(a,b,c) _mm256_fmsub_ps(a,b,c)
#define SIMD_ZERO_ALL() _mm256_zeroall()
#else
#define SIMDf_MUL_ADD(a,b,c) SIMDf_ADD(SIMDf_MUL(a,b),c)
#define SIMDf_MUL_SUB(a,b,c) SIMDf_SUB(SIMDf_MUL(a,b),c)
#define SIMD_ZERO_ALL()
#endif

static bool VAR(SIMD_Values_Set) = false;

static SIMDf SIMDf_NUM(incremental);
static SIMDf SIMDf_NUM(0);
static SIMDf SIMDf_NUM(2);
static SIMDf SIMDf_NUM(6);
static SIMDf SIMDf_NUM(10);
static SIMDf SIMDf_NUM(15);
static SIMDf SIMDf_NUM(32);
static SIMDf SIMDf_NUM(999999);

static SIMDf SIMDf_NUM(0_6);

static SIMDf SIMDf_NUM(F3);
static SIMDf SIMDf_NUM(G3);
static SIMDf SIMDf_NUM(G32);
static SIMDf SIMDf_NUM(G33);
static SIMDf SIMDf_NUM(hash2Float);

static SIMDi SIMDi_NUM(incremental);
static SIMDi SIMDi_NUM(1);
static SIMDi SIMDi_NUM(2);
static SIMDi SIMDi_NUM(4);
static SIMDi SIMDi_NUM(8);
static SIMDi SIMDi_NUM(12);
static SIMDi SIMDi_NUM(14);
static SIMDi SIMDi_NUM(15);
static SIMDi SIMDi_NUM(60493);
static SIMDi SIMDi_NUM(19990303);
static SIMDi SIMDi_NUM(0x40000000);
static SIMDi SIMDi_NUM(0x7fffffff);

static SIMDi SIMDi_NUM(xPrime);
static SIMDi SIMDi_NUM(yPrime);
static SIMDi SIMDi_NUM(zPrime);

void FUNC(InitSIMDValues)()
{
	if (VAR(SIMD_Values_Set))
		return;

	uSIMDf incF;
	uSIMDi incI;
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		incF.a[i] = float(i);
		incI.a[i] = i;
	}
	SIMDf_NUM(incremental) = incF.m;
	SIMDi_NUM(incremental) = incI.m;

	SIMDf_NUM(0) = SIMDf_SET_ZERO();
	SIMDf_NUM(1) = SIMDf_SET(1.0f);
	SIMDf_NUM(2) = SIMDf_SET(2.0f);
	SIMDf_NUM(6) = SIMDf_SET(6.0f);
	SIMDf_NUM(10) = SIMDf_SET(10.0f);
	SIMDf_NUM(15) = SIMDf_SET(15.0f);
	SIMDf_NUM(32) = SIMDf_SET(32.0f);
	SIMDf_NUM(999999) = SIMDf_SET(999999.0f);

	SIMDf_NUM(0_6) = SIMDf_SET(0.6f);

	SIMDf_NUM(F3) = SIMDf_SET(1.f / 3.f);
	SIMDf_NUM(G3) = SIMDf_SET(1.f / 6.f);
	SIMDf_NUM(G32) = SIMDf_SET(2.f / 6.f);
	SIMDf_NUM(G33) = SIMDf_SET(3.f / 6.f);
	SIMDf_NUM(hash2Float) = SIMDf_SET(1.f / 1073741824.f);

	SIMDi_NUM(0) = SIMDi_SET_ZERO();
	SIMDi_NUM(1) = SIMDi_SET(1);
	SIMDi_NUM(2) = SIMDi_SET(2);
	SIMDi_NUM(4) = SIMDi_SET(4);
	SIMDi_NUM(8) = SIMDi_SET(8);
	SIMDi_NUM(12) = SIMDi_SET(12);
	SIMDi_NUM(14) = SIMDi_SET(14);
	SIMDi_NUM(15) = SIMDi_SET(15);
	SIMDi_NUM(60493) = SIMDi_SET(60493);
	SIMDi_NUM(0x40000000) = SIMDi_SET(0x40000000);
	SIMDi_NUM(0x7fffffff) = SIMDi_SET(0x7fffffff);

	SIMDi_NUM(xPrime) = SIMDi_SET(1619);
	SIMDi_NUM(yPrime) = SIMDi_SET(31337);
	SIMDi_NUM(zPrime) = SIMDi_SET(6971);

	SIMDi_NUM(0xffffffff) = SIMDi_EQUAL(SIMDi_SET_ZERO(), SIMDi_SET_ZERO());

	VAR(SIMD_Values_Set) = true;
}

static SIMDf FUNC(Lerp)(const SIMDf& a, const SIMDf& b, const SIMDf& t)
{
	SIMDf r;
	r = SIMDf_SUB(b, a);
	r = SIMDf_MUL_ADD(r, t, a);
	return r;
}

static SIMDf FUNC(InterpQuintic)(const SIMDf& t)
{
	SIMDf r;
	r = SIMDf_MUL_SUB(t, SIMDf_NUM(6), SIMDf_NUM(15));
	r = SIMDf_MUL_ADD(r, t, SIMDf_NUM(10));
	r = SIMDf_MUL(r, t);
	r = SIMDf_MUL(r, t);
	r = SIMDf_MUL(r, t);

	return r;
}

SIMD_LEVEL_CLASS::FASTNOISE_SIMD_CLASS(SIMD_LEVEL)(int seed)
{
	m_seed = seed;
	FUNC(InitSIMDValues)();
	s_currentSIMDLevel = SIMD_LEVEL;
}

float* SIMD_LEVEL_CLASS::GetEmptySet(int size)
{
	// size must be a multiple of VECTOR_SIZE (8)
	if ((size & (VECTOR_SIZE - 1)) != 0)
		throw;

    float* noiseSet;

	//SIMD data has to be aligned
	SIMD_ALIGNED_SET(noiseSet, size);

	return noiseSet;
}

#define FILL_SET(f) \
static SIMDf FUNC(f##Single)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z);\
void SIMD_LEVEL_CLASS::Fill##f##Set(float* floatSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance)\
{\
	SIMD_ZERO_ALL();\
	\
	stepDistance *= m_frequency;\
	\
	SIMDi seed = SIMDi_SET(m_seed);\
	\
	SIMDf step = SIMDf_SET(stepDistance);\
	SIMDf vStep = SIMDf_SET(stepDistance * VECTOR_SIZE);\
	\
	SIMDf yBase = SIMDf_SET(yStart * m_frequency);\
	SIMDf zBase = SIMDf_MUL_ADD(step, SIMDf_NUM(incremental), SIMDf_SET(zStart * m_frequency));\
	\
	SIMDf x = SIMDf_SET(xStart * m_frequency);\
	SIMDf y, z;\
	\
	int index = 0;\
	\
	for (int ix = 0; ix < xSize; ix++)\
	{\
		y = yBase;\
		\
		for (int iy = 0; iy < ySize; iy++)\
		{\
			z = zBase;\
			\
			for (int iz = 0; iz < zSize; iz += VECTOR_SIZE)\
			{\
				SIMDf_STORE(&floatSet[index], FUNC(f##Single)(seed, x, y, z));\
				\
				index += VECTOR_SIZE;\
				\
				z = SIMDf_ADD(z, vStep);\
			}\
			y = SIMDf_ADD(y, step);\
		}\
		x = SIMDf_ADD(x, step);\
	}\
	SIMD_ZERO_ALL();\
}

#define FILL_FRACTAL_SET(f) \
void SIMD_LEVEL_CLASS::Fill##f##FractalSet(float* floatSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance)\
{\
	SIMD_ZERO_ALL();\
	\
	stepDistance *= m_frequency;\
	\
	SIMDi seed = SIMDi_SET(m_seed);\
	SIMDf lacunarity = SIMDf_SET(m_lacunarity);\
	SIMDf gain = SIMDf_SET(m_gain);\
	\
	SIMDf step = SIMDf_SET(stepDistance);\
	SIMDf vStep = SIMDf_SET(stepDistance * VECTOR_SIZE);\
	\
	SIMDf yBase = SIMDf_SET(yStart * m_frequency);\
	SIMDf zBase = SIMDf_MUL_ADD(step, SIMDf_NUM(incremental), SIMDf_SET(zStart * m_frequency));\
	\
	SIMDf x = SIMDf_SET(xStart * m_frequency);\
	SIMDf y, z;\
	\
	int index = 0;\
	\
	switch(m_fractalType)\
	{\
	case FBM:\
		for (int ix = 0; ix < xSize; ix++)\
		{\
			y = yBase;\
			\
			for (int iy = 0; iy < ySize; iy++)\
			{\
				z = zBase;\
				\
				for (int iz = 0; iz < zSize; iz += VECTOR_SIZE)\
				{\
					SIMDi seedF = seed;\
					SIMDf sumF = FUNC(f##Single)(seed, x, y, z);\
					SIMDf maxF = SIMDf_NUM(1);\
					SIMDf ampF = SIMDf_NUM(1);\
					unsigned int octaveIndex = 0;\
					\
					SIMDf xF = x;\
					SIMDf yF = y;\
					SIMDf zF = z;\
					\
					while (++octaveIndex < m_octaves)\
					{\
						xF = SIMDf_MUL(xF, lacunarity);\
						yF = SIMDf_MUL(yF, lacunarity);\
						zF = SIMDf_MUL(zF, lacunarity);\
						seedF = SIMDi_ADD(seedF, SIMDi_NUM(1));\
						\
						ampF = SIMDf_MUL(ampF, gain);\
						maxF = SIMDf_ADD(maxF, ampF);\
						sumF = SIMDf_MUL_ADD(FUNC(f##Single)(seedF, xF, yF, zF), ampF, sumF);\
					}\
					\
					SIMDf_STORE(&floatSet[index], SIMDf_DIV(sumF, maxF));\
					\
					index += VECTOR_SIZE;\
					\
					z = SIMDf_ADD(z, vStep);\
				}\
				y = SIMDf_ADD(y, step);\
			}\
			x = SIMDf_ADD(x, step);\
		}\
		break;\
	\
	case Billow:\
		for (int ix = 0; ix < xSize; ix++)\
		{\
			y = yBase;\
			\
			for (int iy = 0; iy < ySize; iy++)\
			{\
				z = zBase;\
				\
				for (int iz = 0; iz < zSize; iz += VECTOR_SIZE)\
				{\
					SIMDi seedF = seed;\
					SIMDf sumF = SIMDf_MUL_SUB(SIMDf_ABS(FUNC(f##Single)(seed, x, y, z)), SIMDf_NUM(2), SIMDf_NUM(1));\
					SIMDf maxF = SIMDf_NUM(1);\
					SIMDf ampF = SIMDf_NUM(1);\
					unsigned int octaveIndex = 0;\
					\
					SIMDf xF = x;\
					SIMDf yF = y;\
					SIMDf zF = z;\
					\
					while (++octaveIndex < m_octaves)\
					{\
						xF = SIMDf_MUL(xF, lacunarity);\
						yF = SIMDf_MUL(yF, lacunarity);\
						zF = SIMDf_MUL(zF, lacunarity);\
						seedF = SIMDi_ADD(seedF, SIMDi_NUM(1));\
						\
						ampF = SIMDf_MUL(ampF, gain);\
						maxF = SIMDf_ADD(maxF, ampF);\
						sumF = SIMDf_MUL_ADD(SIMDf_MUL_SUB(SIMDf_ABS(FUNC(f##Single)(seedF, xF, yF, zF)), SIMDf_NUM(2), SIMDf_NUM(1)), ampF, sumF);\
					}\
					\
					SIMDf_STORE(&floatSet[index], SIMDf_DIV(sumF, maxF));\
					\
					index += VECTOR_SIZE;\
					\
					z = SIMDf_ADD(z, vStep);\
				}\
				y = SIMDf_ADD(y, step);\
			}\
			x = SIMDf_ADD(x, step);\
		}\
		break;\
	\
	case RigidMulti:\
		for (int ix = 0; ix < xSize; ix++)\
		{\
			y = yBase;\
			\
			for (int iy = 0; iy < ySize; iy++)\
			{\
				z = zBase;\
				\
				for (int iz = 0; iz < zSize; iz += VECTOR_SIZE)\
				{\
					SIMDi seedF = seed;\
					SIMDf sumF = SIMDf_SUB(SIMDf_NUM(1), SIMDf_ABS(FUNC(f##Single)(seed, x, y, z)));\
					SIMDf ampF = SIMDf_NUM(1);\
					unsigned int octaveIndex = 0;\
					\
					SIMDf xF = x;\
					SIMDf yF = y;\
					SIMDf zF = z;\
					\
					while (++octaveIndex < m_octaves)\
					{\
						xF = SIMDf_MUL(xF, lacunarity);\
						yF = SIMDf_MUL(yF, lacunarity);\
						zF = SIMDf_MUL(zF, lacunarity);\
						seedF = SIMDi_ADD(seedF, SIMDi_NUM(1));\
						\
						ampF = SIMDf_MUL(ampF, gain);\
						sumF = SIMDf_SUB(sumF, SIMDf_MUL(SIMDf_SUB(SIMDf_NUM(1), SIMDf_ABS(FUNC(f##Single)(seedF, xF, yF, zF))), ampF));\
					}\
					\
					SIMDf_STORE(&floatSet[index], sumF);\
					\
					index += VECTOR_SIZE;\
					\
					z = SIMDf_ADD(z, vStep);\
				}\
				y = SIMDf_ADD(y, step);\
			}\
			x = SIMDf_ADD(x, step);\
		}\
		break;\
	}\
	SIMD_ZERO_ALL();\
}

FILL_SET(WhiteNoise)

FILL_SET(Value)
FILL_FRACTAL_SET(Value)

FILL_SET(Gradient)
FILL_FRACTAL_SET(Gradient)

FILL_SET(Simplex)
FILL_FRACTAL_SET(Simplex)

static SIMDi FUNC(Hash)(const SIMDi& seed, const SIMDi& x, const SIMDi& y, const SIMDi& z)
{
	SIMDi hash;

	hash = SIMDi_MUL(x, SIMDi_NUM(xPrime));
	hash = SIMDi_ADD(SIMDi_MUL(y, SIMDi_NUM(yPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(z, SIMDi_NUM(zPrime)), hash);
	hash = SIMDi_XOR(hash, seed);
	hash = SIMDi_XOR(SIMDi_SHIFT_R(hash, 13), hash);

	hash = SIMDi_MUL(SIMDi_ADD(SIMDi_MUL(SIMDi_MUL(hash, hash), SIMDi_NUM(60493)), SIMDi_NUM(19990303)), hash);
	hash = SIMDi_XOR(SIMDi_SHIFT_R(hash, 13), hash);

	return hash;
}

static SIMDf FUNC(ValCoord)(const SIMDi& seed, const SIMDi& x, const SIMDi& y, const SIMDi& z)
{
	// High bit hash
	SIMDi hash;

	hash = SIMDi_MUL(x, SIMDi_NUM(xPrime));
	hash = SIMDi_ADD(SIMDi_MUL(y, SIMDi_NUM(yPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(z, SIMDi_NUM(zPrime)), hash);
	hash = SIMDi_XOR(hash, seed);
	hash = SIMDi_XOR(SIMDi_SHIFT_R(hash, 13), hash);

	hash = SIMDi_AND(SIMDi_MUL(SIMDi_ADD(SIMDi_MUL(SIMDi_MUL(hash, hash), SIMDi_NUM(60493)), SIMDi_NUM(19990303)), hash), SIMDi_NUM(0x7fffffff));

	return SIMDf_MUL(SIMDf_NUM(hash2Float), SIMDf_CONVERT_TO_FLOAT(SIMDi_SUB(hash, SIMDi_NUM(0x40000000))));
}

static SIMDf FUNC(GradCoord)(const SIMDi& seed, const SIMDi& xi, const SIMDi& yi, const SIMDi& zi, const SIMDf& x, const SIMDf& y, const SIMDf& z)
{
	SIMDi hash = SIMDi_AND(FUNC(Hash)(seed, xi, yi, zi), SIMDi_NUM(15));

	//if h < 8 then x, else y
	SIMDf u = SIMDf_CAST_TO_FLOAT(SIMDi_LESS_THAN(hash, SIMDi_NUM(8)));
	u = SIMDf_BLENDV(y, x, u);

	//if h < 4 then y else if h is 12 or 14 then x else z
	SIMDf v = SIMDf_CAST_TO_FLOAT(SIMDi_LESS_THAN(hash, SIMDi_NUM(4)));
	SIMDf h12o14 = SIMDf_CAST_TO_FLOAT(SIMDi_OR(SIMDi_EQUAL(hash, SIMDi_NUM(12)), SIMDi_EQUAL(hash, SIMDi_NUM(14))));
	h12o14 = SIMDf_BLENDV(z, x, h12o14);
	v = SIMDf_BLENDV(h12o14, y, v);

	//if h1 then -u else u
	//if h2 then -v else v
	SIMDf h1 = SIMDf_CAST_TO_FLOAT(SIMDi_SHIFT_L(SIMDi_AND(hash, SIMDi_NUM(1)), 31));
	SIMDf h2 = SIMDf_CAST_TO_FLOAT(SIMDi_SHIFT_L(SIMDi_AND(hash, SIMDi_NUM(2)), 30));
	//then add them
	return SIMDf_ADD(SIMDf_BLENDV(u, SIMDf_SUB(SIMDf_NUM(0), u), h1), SIMDf_BLENDV(v, SIMDf_SUB(SIMDf_NUM(0), v), h2));
}

static SIMDf FUNC(WhiteNoiseSingle)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z)
{
	return FUNC(ValCoord)(seed,
		SIMDi_XOR(SIMDi_CAST_TO_INT(x), SIMDi_SHIFT_R(SIMDi_CAST_TO_INT(x), 16)),
		SIMDi_XOR(SIMDi_CAST_TO_INT(y), SIMDi_SHIFT_R(SIMDi_CAST_TO_INT(y), 16)),
		SIMDi_XOR(SIMDi_CAST_TO_INT(z), SIMDi_SHIFT_R(SIMDi_CAST_TO_INT(z), 16)));
}

static SIMDf FUNC(ValueSingle)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z)
{
	SIMDf xs = SIMDf_FLOOR(x);
	SIMDf ys = SIMDf_FLOOR(y);
	SIMDf zs = SIMDf_FLOOR(z);

	SIMDi x0 = SIMDi_CONVERT_TO_INT(xs);
	SIMDi y0 = SIMDi_CONVERT_TO_INT(ys);
	SIMDi z0 = SIMDi_CONVERT_TO_INT(zs);
	SIMDi x1 = SIMDi_ADD(x0, SIMDi_NUM(1));
	SIMDi y1 = SIMDi_ADD(y0, SIMDi_NUM(1));
	SIMDi z1 = SIMDi_ADD(z0, SIMDi_NUM(1));

	xs = FUNC(InterpQuintic)(SIMDf_SUB(x, xs));
	ys = FUNC(InterpQuintic)(SIMDf_SUB(y, ys));
	zs = FUNC(InterpQuintic)(SIMDf_SUB(z, zs));

	return FUNC(Lerp)(
		FUNC(Lerp)(
			FUNC(Lerp)(FUNC(ValCoord)(seed, x0, y0, z0), FUNC(ValCoord)(seed, x1, y0, z0), xs),
			FUNC(Lerp)(FUNC(ValCoord)(seed, x0, y1, z0), FUNC(ValCoord)(seed, x1, y1, z0), xs), ys),
		FUNC(Lerp)(
			FUNC(Lerp)(FUNC(ValCoord)(seed, x0, y0, z1), FUNC(ValCoord)(seed, x1, y0, z1), xs),
			FUNC(Lerp)(FUNC(ValCoord)(seed, x0, y1, z1), FUNC(ValCoord)(seed, x1, y1, z1), xs), ys), zs);
}

static SIMDf FUNC(GradientSingle)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z)
{
	SIMDf xs = SIMDf_FLOOR(x);
	SIMDf ys = SIMDf_FLOOR(y);
	SIMDf zs = SIMDf_FLOOR(z);

	SIMDi x0 = SIMDi_CONVERT_TO_INT(xs);
	SIMDi y0 = SIMDi_CONVERT_TO_INT(ys);
	SIMDi z0 = SIMDi_CONVERT_TO_INT(zs);
	SIMDi x1 = SIMDi_ADD(x0, SIMDi_NUM(1));
	SIMDi y1 = SIMDi_ADD(y0, SIMDi_NUM(1));
	SIMDi z1 = SIMDi_ADD(z0, SIMDi_NUM(1));

	SIMDf xf0 = xs = SIMDf_SUB(x, xs);
	SIMDf yf0 = ys = SIMDf_SUB(y, ys);
	SIMDf zf0 = zs = SIMDf_SUB(z, zs);
	SIMDf xf1 = SIMDf_SUB(xf0, SIMDf_NUM(1));
	SIMDf yf1 = SIMDf_SUB(yf0, SIMDf_NUM(1));
	SIMDf zf1 = SIMDf_SUB(zf0, SIMDf_NUM(1));

	xs = FUNC(InterpQuintic)(xs);
	ys = FUNC(InterpQuintic)(ys);
	zs = FUNC(InterpQuintic)(zs);

	return FUNC(Lerp)(
		FUNC(Lerp)(
			FUNC(Lerp)(FUNC(GradCoord)(seed, x0, y0, z0, xf0, yf0, zf0), FUNC(GradCoord)(seed, x1, y0, z0, xf1, yf0, zf0), xs),
			FUNC(Lerp)(FUNC(GradCoord)(seed, x0, y1, z0, xf0, yf1, zf0), FUNC(GradCoord)(seed, x1, y1, z0, xf1, yf1, zf0), xs), ys),
		FUNC(Lerp)(
			FUNC(Lerp)(FUNC(GradCoord)(seed, x0, y0, z1, xf0, yf0, zf1), FUNC(GradCoord)(seed, x1, y0, z1, xf1, yf0, zf1), xs),
			FUNC(Lerp)(FUNC(GradCoord)(seed, x0, y1, z1, xf0, yf1, zf1), FUNC(GradCoord)(seed, x1, y1, z1, xf1, yf1, zf1), xs), ys), zs);
}

static SIMDf FUNC(SimplexSingle)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z)
{
	SIMDf f = SIMDf_MUL(SIMDf_NUM(F3), SIMDf_ADD(SIMDf_ADD(x, y), z));
	SIMDf x0 = SIMDf_FLOOR(SIMDf_ADD(x, f));
	SIMDf y0 = SIMDf_FLOOR(SIMDf_ADD(y, f));
	SIMDf z0 = SIMDf_FLOOR(SIMDf_ADD(z, f));

	SIMDi i = SIMDi_CONVERT_TO_INT(x0);
	SIMDi j = SIMDi_CONVERT_TO_INT(y0);
	SIMDi k = SIMDi_CONVERT_TO_INT(z0);

	SIMDf g = SIMDf_MUL(SIMDf_NUM(G3), SIMDf_CONVERT_TO_FLOAT(SIMDi_ADD(SIMDi_ADD(i, j), k)));
	x0 = SIMDf_SUB(x, SIMDf_SUB(x0, g));
	y0 = SIMDf_SUB(y, SIMDf_SUB(y0, g));
	z0 = SIMDf_SUB(z, SIMDf_SUB(z0, g));

	SIMDi x0_ge_y0 = SIMDi_CAST_TO_INT(SIMDf_GREATER_EQUAL(x0, y0));
	SIMDi y0_ge_z0 = SIMDi_CAST_TO_INT(SIMDf_GREATER_EQUAL(y0, z0));
	SIMDi x0_ge_z0 = SIMDi_CAST_TO_INT(SIMDf_GREATER_EQUAL(x0, z0));

	SIMDi i1 = SIMDi_AND(SIMDi_NUM(1), SIMDi_AND(x0_ge_y0, x0_ge_z0));
	SIMDi j1 = SIMDi_AND(SIMDi_NUM(1), SIMDi_AND_NOT(x0_ge_y0, y0_ge_z0));
	SIMDi k1 = SIMDi_AND(SIMDi_NUM(1), SIMDi_AND_NOT(x0_ge_z0, SIMDi_NOT(y0_ge_z0)));

	SIMDi i2 = SIMDi_AND(SIMDi_NUM(1), SIMDi_OR(x0_ge_y0, x0_ge_z0));
	SIMDi j2 = SIMDi_AND(SIMDi_NUM(1), SIMDi_OR(SIMDi_NOT(x0_ge_y0), y0_ge_z0));
	SIMDi k2 = SIMDi_AND(SIMDi_NUM(1), SIMDi_NOT(SIMDi_AND(x0_ge_z0, y0_ge_z0)));

	SIMDf x1 = SIMDf_ADD(SIMDf_SUB(x0, SIMDf_CONVERT_TO_FLOAT(i1)), SIMDf_NUM(G3));
	SIMDf y1 = SIMDf_ADD(SIMDf_SUB(y0, SIMDf_CONVERT_TO_FLOAT(j1)), SIMDf_NUM(G3));
	SIMDf z1 = SIMDf_ADD(SIMDf_SUB(z0, SIMDf_CONVERT_TO_FLOAT(k1)), SIMDf_NUM(G3));
	SIMDf x2 = SIMDf_ADD(SIMDf_SUB(x0, SIMDf_CONVERT_TO_FLOAT(i2)), SIMDf_NUM(G32));
	SIMDf y2 = SIMDf_ADD(SIMDf_SUB(y0, SIMDf_CONVERT_TO_FLOAT(j2)), SIMDf_NUM(G32));
	SIMDf z2 = SIMDf_ADD(SIMDf_SUB(z0, SIMDf_CONVERT_TO_FLOAT(k2)), SIMDf_NUM(G32));
	SIMDf x3 = SIMDf_ADD(SIMDf_SUB(x0, SIMDf_NUM(1)), SIMDf_NUM(G33));
	SIMDf y3 = SIMDf_ADD(SIMDf_SUB(y0, SIMDf_NUM(1)), SIMDf_NUM(G33));
	SIMDf z3 = SIMDf_ADD(SIMDf_SUB(z0, SIMDf_NUM(1)), SIMDf_NUM(G33));

	SIMDf t0 = SIMDf_SUB(SIMDf_SUB(SIMDf_SUB(SIMDf_NUM(0_6), SIMDf_MUL(x0, x0)), SIMDf_MUL(y0, y0)), SIMDf_MUL(z0, z0));
	SIMDf t1 = SIMDf_SUB(SIMDf_SUB(SIMDf_SUB(SIMDf_NUM(0_6), SIMDf_MUL(x1, x1)), SIMDf_MUL(y1, y1)), SIMDf_MUL(z1, z1));
	SIMDf t2 = SIMDf_SUB(SIMDf_SUB(SIMDf_SUB(SIMDf_NUM(0_6), SIMDf_MUL(x2, x2)), SIMDf_MUL(y2, y2)), SIMDf_MUL(z2, z2));
	SIMDf t3 = SIMDf_SUB(SIMDf_SUB(SIMDf_SUB(SIMDf_NUM(0_6), SIMDf_MUL(x3, x3)), SIMDf_MUL(y3, y3)), SIMDf_MUL(z3, z3));

	SIMDf n0 = SIMDf_GREATER_EQUAL(t0, SIMDf_NUM(0));
	SIMDf n1 = SIMDf_GREATER_EQUAL(t1, SIMDf_NUM(0));
	SIMDf n2 = SIMDf_GREATER_EQUAL(t2, SIMDf_NUM(0));
	SIMDf n3 = SIMDf_GREATER_EQUAL(t3, SIMDf_NUM(0));

	t0 = SIMDf_MUL(t0, t0);
	t1 = SIMDf_MUL(t1, t1);
	t2 = SIMDf_MUL(t2, t2);
	t3 = SIMDf_MUL(t3, t3);

	n0 = SIMDf_AND(n0, SIMDf_MUL(SIMDf_MUL(t0, t0), FUNC(GradCoord)(seed, i, j, k, x0, y0, z0)));
	n1 = SIMDf_AND(n1, SIMDf_MUL(SIMDf_MUL(t1, t1), FUNC(GradCoord)(seed, SIMDi_ADD(i, i1), SIMDi_ADD(j, j1), SIMDi_ADD(k, k1), x1, y1, z1)));
	n2 = SIMDf_AND(n2, SIMDf_MUL(SIMDf_MUL(t2, t2), FUNC(GradCoord)(seed, SIMDi_ADD(i, i2), SIMDi_ADD(j, j2), SIMDi_ADD(k, k2), x2, y2, z2)));
	n3 = SIMDf_AND(n3, SIMDf_MUL(SIMDf_MUL(t3, t3), FUNC(GradCoord)(seed, SIMDi_ADD(i, SIMDi_NUM(1)), SIMDi_ADD(j, SIMDi_NUM(1)), SIMDi_ADD(k, SIMDi_NUM(1)), x3, y3, z3)));

	return SIMDf_MUL(SIMDf_NUM(32), SIMDf_ADD(n0, SIMDf_ADD(n1, SIMDf_ADD(n2, n3))));
}
#if defined(__GNUC__) && SIMD_LEVEL != FN_NO_SIMD_FALLBACK && SIMD_LEVEL != FN_AVX2
#pragma GCC pop_options
#endif
#undef SIMD_LEVEL
#endif
