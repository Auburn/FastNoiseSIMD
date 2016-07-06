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

#include "FastNoiseSIMD.h"
#include <assert.h> 

#if defined(SIMD_LEVEL) || defined(FN_COMPILE_NO_SIMD_FALLBACK)

#ifndef SIMD_LEVEL
#define SIMD_LEVEL FN_NO_SIMD_FALLBACK
#define SIMD_LEVEL_H FN_NO_SIMD_FALLBACK
#include "FastNoiseSIMD_internal.h"
#include <math.h>
#define FN_ALIGNED_SETS
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
#define SIMD_ALLOCATE_SET(floatP, floatCount) floatP = (float*)_aligned_malloc((floatCount)* sizeof(float), MEMORY_ALIGNMENT)
#else
#define SIMD_ALLOCATE_SET(floatP, floatCount) posix_memalign((void**)&floatP, MEMORY_ALIGNMENT, (floatCount)* sizeof(float))
#endif
#else
#define SIMD_ALLOCATE_SET(floatP, floatCount) floatP = new float[floatCount]
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

static SIMDi SIMDi_NUM(0xffffffff);
static SIMDf SIMDf_NUM(1);

// SIMD functions
#if SIMD_LEVEL >= FN_AVX2

#ifdef FN_ALIGNED_SETS
#define SIMDf_STORE(p,a) _mm256_store_ps(p,a)
#define SIMDf_LOAD(p) _mm256_load_ps(p)
#else
#define SIMDf_STORE(p,a) _mm256_storeu_ps(p,a)
#define SIMDf_LOAD(p) _mm256_loadu_ps(p)
#endif

#define SIMDf_ADD(a,b) _mm256_add_ps(a,b)
#define SIMDf_SUB(a,b) _mm256_sub_ps(a,b)
#define SIMDf_MUL(a,b) _mm256_mul_ps(a,b)
#define SIMDf_DIV(a,b) _mm256_div_ps(a,b)

#define SIMDf_MIN(a,b) _mm256_min_ps(a,b)
#define SIMDf_MAX(a,b) _mm256_max_ps(a,b)
#define SIMDf_INV_SQRT(a) _mm256_rsqrt_ps(a)

#define SIMDf_LESS_THAN(a,b) _mm256_cmp_ps(a,b,_CMP_LT_OQ)
#define SIMDf_GREATER_THAN(a,b) _mm256_cmp_ps(a,b,_CMP_GT_OQ)
#define SIMDf_LESS_EQUAL(a,b) _mm256_cmp_ps(a,b,_CMP_LE_OQ)
#define SIMDf_GREATER_EQUAL(a,b) _mm256_cmp_ps(a,b,_CMP_GE_OQ)

#define SIMDf_AND(a,b) _mm256_and_ps(a,b)
#define SIMDf_AND_NOT(a,b) _mm256_andnot_ps(a,b)
#define SIMDf_XOR(a,b) _mm256_xor_ps(a,b)

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

#define SIMDi_SHIFT_R(a, b) _mm256_srai_epi32(a, b)
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
#define SIMDf_LOAD(p) _mm_load_ps(p)
#else
#define SIMDf_STORE(p,a) _mm_storeu_ps(p,a)
#define SIMDf_LOAD(p) _mm_loadu_ps(p)
#endif

#define SIMDf_ADD(a,b) _mm_add_ps(a,b)
#define SIMDf_SUB(a,b) _mm_sub_ps(a,b)
#define SIMDf_MUL(a,b) _mm_mul_ps(a,b)
#define SIMDf_DIV(a,b) _mm_div_ps(a,b)

#define SIMDf_MIN(a,b) _mm_min_ps(a,b)
#define SIMDf_MAX(a,b) _mm_max_ps(a,b)
#define SIMDf_INV_SQRT(a) _mm_rsqrt_ps(a)

#define SIMDf_LESS_THAN(a,b) _mm_cmplt_ps(a,b)
#define SIMDf_GREATER_THAN(a,b) _mm_cmpgt_ps(a,b)
#define SIMDf_LESS_EQUAL(a,b) _mm_cmple_ps(a,b)
#define SIMDf_GREATER_EQUAL(a,b) _mm_cmpge_ps(a,b)

#define SIMDf_AND(a,b) _mm_and_ps(a,b)
#define SIMDf_AND_NOT(a,b) _mm_andnot_ps(a,b)
#define SIMDf_XOR(a,b) _mm_xor_ps(a,b)

#if SIMD_LEVEL == FN_SSE41
#define SIMDi_MUL(a,b) _mm_mullo_epi32(a,b)
#define SIMDf_FLOOR(a) _mm_floor_ps(a)
#define SIMDf_BLENDV(a,b,mask) _mm_blendv_ps(a,b,mask)
#else
static SIMDi FUNC(MUL)(const SIMDi& a, const SIMDi& b)
{
	__m128 tmp1 = _mm_castsi128_ps(_mm_mul_epu32(a, b)); /* mul 2,0*/
	__m128 tmp2 = _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4))); /* mul 3,1 */
	return _mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(2, 0, 2, 0))), _MM_SHUFFLE(3, 1, 2, 0));
}
#define SIMDi_MUL(a,b) FUNC(MUL)(a,b)

static SIMDf FUNC(FLOOR)(const SIMDf& a)
{
	__m128 fval = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));

	return _mm_sub_ps(fval, _mm_and_ps(_mm_cmplt_ps(a, fval), SIMDf_NUM(1)));
}
#define SIMDf_FLOOR(a) FUNC(FLOOR)(a)

#define SIMDf_BLENDV(a,b,mask) _mm_or_ps(_mm_andnot_ps(mask, a), _mm_and_ps(mask, b))
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

#define SIMDi_SHIFT_R(a,b) _mm_srai_epi32(a, b)
#define SIMDi_SHIFT_L(a,b) _mm_slli_epi32(a, b)

#define SIMDi_EQUAL(a,b) _mm_cmpeq_epi32(a,b)
#define SIMDi_GREATER_THAN(a,b) _mm_cmpgt_epi32(a,b)
#define SIMDi_LESS_THAN(a,b) _mm_cmpgt_epi32(b,a)

#define SIMDf_CONVERT_TO_FLOAT(a) _mm_cvtepi32_ps(a)
#define SIMDf_CAST_TO_FLOAT(a) _mm_castsi128_ps(a)
#define SIMDi_CONVERT_TO_INT(a) _mm_cvtps_epi32(a)
#define SIMDi_CAST_TO_INT(a) _mm_castps_si128(a)

#else // Fallback

static int FUNC(CAST_TO_INT)(float f) { return *reinterpret_cast<int*>(&f); }
static float FUNC(CAST_TO_FLOAT)(int i) { return *reinterpret_cast<float*>(&i); }
#define SIMDi_CAST_TO_INT(a) FUNC(CAST_TO_INT)(a)
#define SIMDf_CAST_TO_FLOAT(a) FUNC(CAST_TO_FLOAT)(a)

#define SIMDf_STORE(p,a) (*(p) = a)
#define SIMDf_LOAD(p) (*p)

#define SIMDf_ADD(a,b) ((a) + (b))
#define SIMDf_SUB(a,b) ((a) - (b))
#define SIMDf_MUL(a,b) ((a) * (b))
#define SIMDf_DIV(a,b) ((a) / (b))

#define SIMDf_MIN(a,b) fminf(a,b)
#define SIMDf_MAX(a,b) fmaxf(a,b)

static float FUNC(INV_SQRT)(float x)
{
	float xhalf = 0.5f * x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i >> 1);
	x = *(float*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
}
#define SIMDf_INV_SQRT(a) FUNC(INV_SQRT)(a)

#define SIMDf_LESS_THAN(a,b) (((a) < (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)
#define SIMDf_GREATER_THAN(a,b) (((a) > (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)
#define SIMDf_LESS_EQUAL(a,b) (((a) <= (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)
#define SIMDf_GREATER_EQUAL(a,b) (((a) >= (b)) ? SIMDf_CAST_TO_FLOAT(0xFFFFFFFF) : 0)

#define SIMDf_AND(a,b) SIMDf_CAST_TO_FLOAT(SIMDi_CAST_TO_INT(a) & SIMDi_CAST_TO_INT(b))
#define SIMDf_AND_NOT(a,b) SIMDf_CAST_TO_FLOAT(~SIMDi_CAST_TO_INT(a) & SIMDi_CAST_TO_INT(b))
#define SIMDf_XOR(a,b) SIMDf_CAST_TO_FLOAT(SIMDi_CAST_TO_INT(a) ^ SIMDi_CAST_TO_INT(b))

#define SIMDf_FLOOR(a) floorf(a)
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

#define SIMDi_CONVERT_TO_INT(a) static_cast<int>(roundf(a))
#define SIMDf_CONVERT_TO_FLOAT(a) static_cast<float>(a)
#endif

//#define SIMDf_SIGN_FLIP(a) SIMDf_XOR(a,SIMDf_NUM(neg0)))
//#define SIMDi_GREATER_EQUAL(a,b) SIMDi_NOT(SIMDi_LESS_THAN(a,b))
//#define SIMDi_LESS_EQUAL(a,b) SIMDi_NOT(SIMDi_GREATER_THAN(a,b))
//#define SIMDi_BLENDV(a,b, mask) SIMDi_CAST_TO_INT(SIMDf_BLENDV(SIMDf_CAST_TO_FLOAT(a),SIMDf_CAST_TO_FLOAT(b),SIMDf_CAST_TO_FLOAT(mask)))

#if SIMD_LEVEL == FN_NO_SIMD_FALLBACK
#define SIMDf_ABS(a) fabsf(a)
#else
#define SIMDf_ABS(a) SIMDf_AND(a,SIMDf_CAST_TO_FLOAT(SIMDi_NUM(0x7fffffff)))
#endif

#if SIMD_LEVEL == FN_AVX2
#define SIMD_ZERO_ALL() _mm256_zeroall()
#else
#define SIMD_ZERO_ALL()
#endif

// FMA3
#if defined(FN_USE_FMA3) && SIMD_LEVEL == FN_AVX2
#define SIMDf_MUL_ADD(a,b,c) _mm256_fmadd_ps(a,b,c)
#define SIMDf_MUL_SUB(a,b,c) _mm256_fmsub_ps(a,b,c)
#else
#define SIMDf_MUL_ADD(a,b,c) SIMDf_ADD(SIMDf_MUL(a,b),c)
#define SIMDf_MUL_SUB(a,b,c) SIMDf_SUB(SIMDf_MUL(a,b),c)
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

static SIMDf SIMDf_NUM(0_45);
static SIMDf SIMDf_NUM(0_5);
static SIMDf SIMDf_NUM(0_6);
static SIMDf SIMDf_NUM(511_5);

static SIMDf SIMDf_NUM(F3);
static SIMDf SIMDf_NUM(G3);
static SIMDf SIMDf_NUM(G32);
static SIMDf SIMDf_NUM(G33);
static SIMDf SIMDf_NUM(hash2Float);
static SIMDf SIMDf_NUM(vectorSize);

static SIMDi SIMDi_NUM(incremental);
static SIMDi SIMDi_NUM(1);
static SIMDi SIMDi_NUM(2);
static SIMDi SIMDi_NUM(3);
static SIMDi SIMDi_NUM(4);
static SIMDi SIMDi_NUM(8);
static SIMDi SIMDi_NUM(12);
static SIMDi SIMDi_NUM(14);
static SIMDi SIMDi_NUM(15);
static SIMDi SIMDi_NUM(255);
static SIMDi SIMDi_NUM(60493);
static SIMDi SIMDi_NUM(0x40000000);
static SIMDi SIMDi_NUM(0x7fffffff);

static SIMDi SIMDi_NUM(xPrime);
static SIMDi SIMDi_NUM(yPrime);
static SIMDi SIMDi_NUM(zPrime);
static SIMDi SIMDi_NUM(bit10Mask);
static SIMDi SIMDi_NUM(vectorSize);

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

	SIMDf_NUM(0_45) = SIMDf_SET(0.45f);
	SIMDf_NUM(0_5) = SIMDf_SET(0.5f);
	SIMDf_NUM(0_6) = SIMDf_SET(0.6f);
	SIMDf_NUM(511_5) = SIMDf_SET(511.5f);

	SIMDf_NUM(F3) = SIMDf_SET(1.f / 3.f);
	SIMDf_NUM(G3) = SIMDf_SET(1.f / 6.f);
	SIMDf_NUM(G32) = SIMDf_SET(2.f / 6.f);
	SIMDf_NUM(G33) = SIMDf_SET(3.f / 6.f);
	SIMDf_NUM(hash2Float) = SIMDf_SET(1.f / 1073741824.f);
	SIMDf_NUM(vectorSize) = SIMDf_SET(VECTOR_SIZE);

	SIMDi_NUM(1) = SIMDi_SET(1);
	SIMDi_NUM(2) = SIMDi_SET(2);
	SIMDi_NUM(3) = SIMDi_SET(3);
	SIMDi_NUM(4) = SIMDi_SET(4);
	SIMDi_NUM(8) = SIMDi_SET(8);
	SIMDi_NUM(12) = SIMDi_SET(12);
	SIMDi_NUM(14) = SIMDi_SET(14);
	SIMDi_NUM(15) = SIMDi_SET(15);
	SIMDi_NUM(255) = SIMDi_SET(255);
	SIMDi_NUM(60493) = SIMDi_SET(60493);
	SIMDi_NUM(0x40000000) = SIMDi_SET(0x40000000);
	SIMDi_NUM(0x7fffffff) = SIMDi_SET(0x7fffffff);

	SIMDi_NUM(xPrime) = SIMDi_SET(1619);
	SIMDi_NUM(yPrime) = SIMDi_SET(31337);
	SIMDi_NUM(zPrime) = SIMDi_SET(6971);
	SIMDi_NUM(bit10Mask) = SIMDi_SET(1023);
	SIMDi_NUM(vectorSize) = SIMDi_SET(VECTOR_SIZE);

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

//static SIMDf FUNC(InterpHermite)(const SIMDf& t)
//{
//	SIMDf r;
//	r = SIMDf_MUL(t, SIMDf_NUM(2));
//	r = SIMDf_SUB(SIMDf_ADD(SIMDf_NUM(1), SIMDf_NUM(2)), r);
//	r = SIMDf_MUL(r, t);
//	r = SIMDf_MUL(r, t);
//
//	return r;
//}

static SIMDi FUNC(Hash)(const SIMDi& seed, const SIMDi& x, const SIMDi& y, const SIMDi& z)
{
	SIMDi hash = seed;

	hash = SIMDi_ADD(SIMDi_MUL(x, SIMDi_NUM(xPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(y, SIMDi_NUM(yPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(z, SIMDi_NUM(zPrime)), hash);

	hash = SIMDi_MUL(SIMDi_MUL(SIMDi_MUL(hash, hash), SIMDi_NUM(60493)), hash);
	hash = SIMDi_XOR(SIMDi_SHIFT_R(hash, 13), hash);

	return hash;
}

static SIMDi FUNC(HashHB)(const SIMDi& seed, const SIMDi& x, const SIMDi& y, const SIMDi& z)
{
	SIMDi hash = seed;

	hash = SIMDi_ADD(SIMDi_MUL(x, SIMDi_NUM(xPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(y, SIMDi_NUM(yPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(z, SIMDi_NUM(zPrime)), hash);
	hash = SIMDi_XOR(SIMDi_SHIFT_R(hash, 13), hash);

	hash = SIMDi_MUL(SIMDi_MUL(SIMDi_MUL(hash, hash), SIMDi_NUM(60493)), hash);

	return hash;
}

static SIMDf FUNC(ValCoord)(const SIMDi& seed, const SIMDi& x, const SIMDi& y, const SIMDi& z)
{
	// High bit hash
	SIMDi hash = seed;

	hash = SIMDi_ADD(SIMDi_MUL(x, SIMDi_NUM(xPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(y, SIMDi_NUM(yPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(z, SIMDi_NUM(zPrime)), hash);
	hash = SIMDi_XOR(SIMDi_SHIFT_R(hash, 13), hash);

	hash = SIMDi_AND(SIMDi_MUL(SIMDi_MUL(SIMDi_MUL(hash, hash), SIMDi_NUM(60493)), hash), SIMDi_NUM(0x7fffffff));

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
	return SIMDf_ADD(SIMDf_XOR(u, h1), SIMDf_XOR(v, h2));
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

#define Euclidean_DISTANCE(_x, _y, _z) SIMDf_MUL_ADD(_x, _x, SIMDf_MUL_ADD(_y, _y, SIMDf_MUL(_z, _z)))
#define Manhattan_DISTANCE(_x, _y, _z) SIMDf_ADD(SIMDf_ADD(SIMDf_ABS(_x), SIMDf_ABS(_y)), SIMDf_ABS(_z))
#define Natural_DISTANCE(_x, _y, _z) SIMDf_ADD(Euclidean_DISTANCE(_x,_y,_z), Manhattan_DISTANCE(_x,_y,_z))

#define Distance2_RETURN(_distance, _distance2) (_distance2)
#define Distance2Add_RETURN(_distance, _distance2) SIMDf_ADD(_distance, _distance2)
#define Distance2Sub_RETURN(_distance, _distance2) SIMDf_SUB(_distance2, _distance)
#define Distance2Mul_RETURN(_distance, _distance2) SIMDf_MUL(_distance, _distance2)
#define Distance2Div_RETURN(_distance, _distance2) SIMDf_DIV(_distance, _distance2)

#define CELLULAR_VALUE_SINGLE(distanceFunc)\
static SIMDf FUNC(CellularValue##distanceFunc##Single)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z)\
{\
	SIMDf distance = SIMDf_NUM(999999);\
	SIMDf cellValue = SIMDf_NUM(0);\
	\
	SIMDi xc = SIMDi_SUB(SIMDi_CONVERT_TO_INT(x), SIMDi_NUM(1));\
	SIMDi ycBase = SIMDi_SUB(SIMDi_CONVERT_TO_INT(y), SIMDi_NUM(1));\
	SIMDi zcBase = SIMDi_SUB(SIMDi_CONVERT_TO_INT(z), SIMDi_NUM(1));\
	\
	for (int xi = 0; xi < 3; xi++)\
	{\
		SIMDf xcf = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(xc), x);\
		SIMDi yc = ycBase;\
		for (int yi = 0; yi < 3; yi++)\
		{\
			SIMDf ycf = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(yc), y);\
			SIMDi zc = zcBase;\
			for (int zi = 0; zi < 3; zi++)\
			{\
				SIMDi hash = FUNC(HashHB)(seed, xc, yc, zc);\
				SIMDf xd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(hash, SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				SIMDf yd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(SIMDi_SHIFT_R(hash,10), SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				SIMDf zd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(SIMDi_SHIFT_R(hash,20), SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				\
				SIMDf invMag = SIMDf_MUL(SIMDf_NUM(0_45), SIMDf_INV_SQRT(SIMDf_MUL_ADD(xd, xd, SIMDf_MUL_ADD(yd, yd, SIMDf_MUL(zd, zd)))));\
				\
				xd = SIMDf_MUL_ADD(xd, invMag, xcf);\
				yd = SIMDf_MUL_ADD(yd, invMag, ycf);\
				zd = SIMDf_MUL_ADD(zd, invMag, SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(zc), z));\
				\
				SIMDf newCellValue = SIMDf_MUL(SIMDf_NUM(hash2Float), SIMDf_CONVERT_TO_FLOAT(SIMDi_SUB(SIMDi_AND(hash,SIMDi_NUM(0x7fffffff)), SIMDi_NUM(0x40000000))));\
				SIMDf newDistance = distanceFunc##_DISTANCE(xd, yd, zd);\
				\
				SIMDf closer = SIMDf_LESS_THAN(newDistance, distance);\
				\
				distance = SIMDf_MIN(newDistance, distance);\
				cellValue = SIMDf_BLENDV(cellValue, newCellValue, closer);\
				\
				zc = SIMDi_ADD(zc, SIMDi_NUM(1));\
			}\
			yc = SIMDi_ADD(yc, SIMDi_NUM(1));\
		}\
		xc = SIMDi_ADD(xc, SIMDi_NUM(1));\
	}\
	\
	return cellValue;\
}

#define CELLULAR_DISTANCE_SINGLE(distanceFunc)\
static SIMDf FUNC(CellularDistance##distanceFunc##Single)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z)\
{\
	SIMDf distance = SIMDf_NUM(999999);\
	\
	SIMDi xc = SIMDi_SUB(SIMDi_CONVERT_TO_INT(x), SIMDi_NUM(1));\
	SIMDi ycBase = SIMDi_SUB(SIMDi_CONVERT_TO_INT(y), SIMDi_NUM(1));\
	SIMDi zcBase = SIMDi_SUB(SIMDi_CONVERT_TO_INT(z), SIMDi_NUM(1));\
	\
	for (int xi = 0; xi < 3; xi++)\
	{\
		SIMDf xcf = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(xc), x);\
		SIMDi yc = ycBase;\
		for (int yi = 0; yi < 3; yi++)\
		{\
			SIMDf ycf = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(yc), y);\
			SIMDi zc = zcBase;\
			for (int zi = 0; zi < 3; zi++)\
			{\
				SIMDi hash = FUNC(HashHB)(seed, xc, yc, zc);\
				SIMDf xd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(hash, SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				SIMDf yd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(SIMDi_SHIFT_R(hash,10), SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				SIMDf zd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(SIMDi_SHIFT_R(hash,20), SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				\
				SIMDf invMag = SIMDf_MUL(SIMDf_NUM(0_45), SIMDf_INV_SQRT(SIMDf_MUL_ADD(xd, xd, SIMDf_MUL_ADD(yd, yd, SIMDf_MUL(zd, zd)))));\
				\
				xd = SIMDf_MUL_ADD(xd, invMag, xcf);\
				yd = SIMDf_MUL_ADD(yd, invMag, ycf);\
				zd = SIMDf_MUL_ADD(zd, invMag, SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(zc), z));\
				\
				SIMDf newDistance = distanceFunc##_DISTANCE(xd, yd, zd);\
				\
				distance = SIMDf_MIN(distance, newDistance);\
				\
				zc = SIMDi_ADD(zc, SIMDi_NUM(1));\
			}\
			yc = SIMDi_ADD(yc, SIMDi_NUM(1));\
		}\
		xc = SIMDi_ADD(xc, SIMDi_NUM(1));\
	}\
	\
	return SIMDf_SUB(distance, SIMDf_NUM(1));\
}

#define CELLULAR_DISTANCE2_SINGLE(distanceFunc, returnFunc)\
static SIMDf FUNC(Cellular##returnFunc##distanceFunc##Single)(const SIMDi& seed, const SIMDf& x, const SIMDf& y, const SIMDf& z)\
{\
	SIMDf distance = SIMDf_NUM(999999);\
	SIMDf distance2 = SIMDf_NUM(999999);\
	\
	SIMDi xc = SIMDi_SUB(SIMDi_CONVERT_TO_INT(x), SIMDi_NUM(1));\
	SIMDi ycBase = SIMDi_SUB(SIMDi_CONVERT_TO_INT(y), SIMDi_NUM(1));\
	SIMDi zcBase = SIMDi_SUB(SIMDi_CONVERT_TO_INT(z), SIMDi_NUM(1));\
	\
	for (int xi = 0; xi < 3; xi++)\
	{\
		SIMDf xcf = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(xc), x);\
		SIMDi yc = ycBase;\
		for (int yi = 0; yi < 3; yi++)\
		{\
			SIMDf ycf = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(yc), y);\
			SIMDi zc = zcBase;\
			for (int zi = 0; zi < 3; zi++)\
			{\
				SIMDi hash = FUNC(HashHB)(seed, xc, yc, zc);\
				SIMDf xd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(hash, SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				SIMDf yd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(SIMDi_SHIFT_R(hash,10), SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				SIMDf zd = SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(SIMDi_AND(SIMDi_SHIFT_R(hash,20), SIMDi_NUM(bit10Mask))), SIMDf_NUM(511_5));\
				\
				SIMDf invMag = SIMDf_MUL(SIMDf_NUM(0_45), SIMDf_INV_SQRT(SIMDf_MUL_ADD(xd, xd, SIMDf_MUL_ADD(yd, yd, SIMDf_MUL(zd, zd)))));\
				\
				xd = SIMDf_MUL_ADD(xd, invMag, xcf);\
				yd = SIMDf_MUL_ADD(yd, invMag, ycf);\
				zd = SIMDf_MUL_ADD(zd, invMag, SIMDf_SUB(SIMDf_CONVERT_TO_FLOAT(zc), z));\
				\
				SIMDf newDistance = distanceFunc##_DISTANCE(xd, yd, zd);\
				\
				distance2 = SIMDf_MAX(SIMDf_MIN(distance2, newDistance), distance);\
				distance = SIMDf_MIN(distance, newDistance);\
				\
				zc = SIMDi_ADD(zc, SIMDi_NUM(1));\
			}\
			yc = SIMDi_ADD(yc, SIMDi_NUM(1));\
		}\
		xc = SIMDi_ADD(xc, SIMDi_NUM(1));\
	}\
	\
	return SIMDf_SUB(returnFunc##_RETURN(distance, distance2), SIMDf_NUM(1));\
}

CELLULAR_VALUE_SINGLE(Euclidean)
CELLULAR_VALUE_SINGLE(Manhattan)
CELLULAR_VALUE_SINGLE(Natural)

#undef Natural_DISTANCE
#define Natural_DISTANCE(_x, _y, _z) SIMDf_MUL(Euclidean_DISTANCE(_x,_y,_z), Manhattan_DISTANCE(_x,_y,_z))

CELLULAR_DISTANCE_SINGLE(Euclidean)
CELLULAR_DISTANCE_SINGLE(Manhattan)
CELLULAR_DISTANCE_SINGLE(Natural)

#define CELLULAR_DISTANCE2_MULTI(returnFunc)\
CELLULAR_DISTANCE2_SINGLE(Euclidean, returnFunc)\
CELLULAR_DISTANCE2_SINGLE(Manhattan, returnFunc)\
CELLULAR_DISTANCE2_SINGLE(Natural, returnFunc)

CELLULAR_DISTANCE2_MULTI(Distance2)
CELLULAR_DISTANCE2_MULTI(Distance2Add)
CELLULAR_DISTANCE2_MULTI(Distance2Sub)
CELLULAR_DISTANCE2_MULTI(Distance2Div)
CELLULAR_DISTANCE2_MULTI(Distance2Mul)

SIMD_LEVEL_CLASS::FASTNOISE_SIMD_CLASS(SIMD_LEVEL)(int seed)
{
	m_seed = seed;
	FUNC(InitSIMDValues)();
	s_currentSIMDLevel = SIMD_LEVEL;
}

int SIMD_LEVEL_CLASS::AlignedSize(int size)
{
#ifdef FN_ALIGNED_SETS
	// size must be a multiple of VECTOR_SIZE (8)
	if ((size & (VECTOR_SIZE - 1)) != 0)
	{
		size &= ~(VECTOR_SIZE - 1);
		size += VECTOR_SIZE;
	}
#endif
	return size;
}

float* SIMD_LEVEL_CLASS::GetEmptySet(int size)
{
	size = AlignedSize(size);

	float* noiseSet;
	SIMD_ALLOCATE_SET(noiseSet, size);

	return noiseSet;
}

#if defined(FN_MIN_Z_4) || !defined(FN_COMPILE_AVX2)
#define Z_SIZE_ASSERT(_zSize) assert(_zSize >= 4)
#else
#define Z_SIZE_ASSERT(_zSize) assert(_zSize >= 8)
#endif

#if SIMD_LEVEL == FN_AVX2 && defined(FN_MIN_Z_4)
#define AVX_DOUBLE_RESET {\
SIMDi _zReset = SIMDi_GREATER_THAN(z, zEndV); \
y = SIMDi_ADD(y, SIMDi_AND(SIMDi_NUM(1), _zReset)); \
z = SIMDi_SUB(z, SIMDi_AND(zSizeV, _zReset)); \
\
SIMDi _yReset = SIMDi_GREATER_THAN(y, yEndV); \
x = SIMDi_ADD(x, SIMDi_AND(SIMDi_NUM(1), _yReset)); \
y = SIMDi_SUB(y, SIMDi_AND(ySizeV, _yReset));}

#else
#define AVX_DOUBLE_RESET
#endif

#ifdef FN_ALIGNED_SETS
#define STORE_LAST_RESULT(_dest, _source) SIMDf_STORE(_dest, _source)
#else
#include <cstring>
#define STORE_LAST_RESULT(_dest, _source) std::memcpy(_dest, &_source, (maxIndex - index) * 4)
#endif

#define SET_BUILDER(f)\
if ((zSize & (VECTOR_SIZE - 1)) == 0)\
{\
	SIMDi yBase = SIMDi_SET(yStart);\
	SIMDi zBase = SIMDi_ADD(SIMDi_NUM(incremental), SIMDi_SET(zStart));\
	\
	SIMDi x = SIMDi_SET(xStart);\
	\
	int index = 0;\
	\
	for (int ix = 0; ix < xSize; ix++)\
	{\
		SIMDf xf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(x), xFreqV);\
		SIMDi y = yBase;\
		\
		for (int iy = 0; iy < ySize; iy++)\
		{\
			SIMDf yf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(y), yFreqV);\
			SIMDi z = zBase;\
			SIMDf zf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(z), zFreqV);\
			\
			f;\
			SIMDf_STORE(&noiseSet[index], result);\
			\
			int iz = VECTOR_SIZE;\
			while (iz < zSize)\
			{\
				z = SIMDi_ADD(z, SIMDi_NUM(vectorSize));\
				index += VECTOR_SIZE;\
				iz += VECTOR_SIZE;\
				zf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(z), zFreqV);\
				\
				f;\
				SIMDf_STORE(&noiseSet[index], result);\
			}\
			index += VECTOR_SIZE;\
			y = SIMDi_ADD(y, SIMDi_NUM(1));\
		}\
		x = SIMDi_ADD(x, SIMDi_NUM(1));\
	}\
}\
else\
{\
	\
	SIMDi ySizeV = SIMDi_SET(ySize); \
	SIMDi zSizeV = SIMDi_SET(zSize); \
	\
	SIMDi yEndV = SIMDi_SET(yStart + ySize - 1); \
	SIMDi zEndV = SIMDi_SET(zStart + zSize - 1); \
	\
	SIMDi x = SIMDi_SET(xStart); \
	SIMDi y = SIMDi_SET(yStart); \
	SIMDi z = SIMDi_ADD(SIMDi_SET(zStart), SIMDi_NUM(incremental)); \
	AVX_DOUBLE_RESET;\
	\
	int index = 0; \
	int maxIndex = xSize * ySize * zSize; \
	\
	for (; index < maxIndex - VECTOR_SIZE; index += VECTOR_SIZE)\
	{\
		SIMDf xf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(x), xFreqV);\
		SIMDf yf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(y), yFreqV);\
		SIMDf zf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(z), zFreqV);\
		\
		f;\
		SIMDf_STORE(&noiseSet[index], result);\
		\
		z = SIMDi_ADD(z, SIMDi_NUM(vectorSize));\
		\
		SIMDi zReset = SIMDi_GREATER_THAN(z, zEndV);\
		y = SIMDi_ADD(y, SIMDi_AND(SIMDi_NUM(1), zReset));\
		z = SIMDi_SUB(z, SIMDi_AND(zSizeV, zReset));\
		\
		SIMDi yReset = SIMDi_GREATER_THAN(y, yEndV);\
		x = SIMDi_ADD(x, SIMDi_AND(SIMDi_NUM(1), yReset));\
		y = SIMDi_SUB(y, SIMDi_AND(ySizeV, yReset));\
		\
		AVX_DOUBLE_RESET;\
	}\
	\
	SIMDf xf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(x), xFreqV);\
	SIMDf yf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(y), yFreqV);\
	SIMDf zf = SIMDf_MUL(SIMDf_CONVERT_TO_FLOAT(z), zFreqV);\
	\
	f;\
	STORE_LAST_RESULT(&noiseSet[index], result);\
}

// FBM SINGLE
#define FBM_SINGLE(f)\
	SIMDi seedF = seedV;\
	SIMDf xF = xf;\
	SIMDf yF = yf;\
	SIMDf zF = zf;\
	\
	SIMDf result = FUNC(f##Single)(seedF, xF, yF, zF);\
	\
	SIMDf maxF = SIMDf_NUM(1);\
	SIMDf ampF = SIMDf_NUM(1);\
	unsigned int octaveIndex = 0;\
	\
	while (++octaveIndex < m_octaves)\
	{\
		xF = SIMDf_MUL(xF, lacunarityV);\
		yF = SIMDf_MUL(yF, lacunarityV);\
		zF = SIMDf_MUL(zF, lacunarityV);\
		seedF = SIMDi_ADD(seedF, SIMDi_NUM(1));\
		\
		ampF = SIMDf_MUL(ampF, gainV);\
		maxF = SIMDf_ADD(maxF, ampF);\
		result = SIMDf_MUL_ADD(FUNC(f##Single)(seedF, xF, yF, zF), ampF, result);\
	}\
	result = SIMDf_DIV(result, maxF)

// BILLOW SINGLE
#define BILLOW_SINGLE(f)\
	SIMDi seedF = seedV;\
	SIMDf xF = xf;\
	SIMDf yF = yf;\
	SIMDf zF = zf;\
	\
	SIMDf result = SIMDf_MUL_SUB(SIMDf_ABS(FUNC(f##Single)(seedF, xF, yF, zF)), SIMDf_NUM(2), SIMDf_NUM(1));\
	\
	SIMDf maxF = SIMDf_NUM(1);\
	SIMDf ampF = SIMDf_NUM(1);\
	unsigned int octaveIndex = 0;\
	\
	while (++octaveIndex < m_octaves)\
	{\
		xF = SIMDf_MUL(xF, lacunarityV);\
		yF = SIMDf_MUL(yF, lacunarityV);\
		zF = SIMDf_MUL(zF, lacunarityV);\
		seedF = SIMDi_ADD(seedF, SIMDi_NUM(1));\
		\
		ampF = SIMDf_MUL(ampF, gainV);\
		maxF = SIMDf_ADD(maxF, ampF);\
		result = SIMDf_MUL_ADD(SIMDf_MUL_SUB(SIMDf_ABS(FUNC(f##Single)(seedF, xF, yF, zF)), SIMDf_NUM(2), SIMDf_NUM(1)), ampF, result);\
	}\
	result = SIMDf_DIV(result, maxF)

// RIGIDMULTI SINGLE
#define RIGIDMULTI_SINGLE(f)\
	SIMDi seedF = seedV;\
	SIMDf xF = xf;\
	SIMDf yF = yf;\
	SIMDf zF = zf;\
	\
	SIMDf result = SIMDf_SUB(SIMDf_NUM(1), SIMDf_ABS(FUNC(f##Single)(seedF, xF, yF, zF)));\
	\
	SIMDf ampF = SIMDf_NUM(1);\
	unsigned int octaveIndex = 0;\
	\
	while (++octaveIndex < m_octaves)\
	{\
		xF = SIMDf_MUL(xF, lacunarityV);\
		yF = SIMDf_MUL(yF, lacunarityV);\
		zF = SIMDf_MUL(zF, lacunarityV);\
		seedF = SIMDi_ADD(seedF, SIMDi_NUM(1));\
		\
		ampF = SIMDf_MUL(ampF, gainV);\
		result = SIMDf_SUB(result, SIMDf_MUL(SIMDf_SUB(SIMDf_NUM(1), SIMDf_ABS(FUNC(f##Single)(seedF, xF, yF, zF))), ampF));\
	}

#define FILL_SET(func) \
void SIMD_LEVEL_CLASS::Fill##func##Set(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)\
{\
	assert(noiseSet);\
	Z_SIZE_ASSERT(zSize);\
	SIMD_ZERO_ALL();\
	SIMDi seedV = SIMDi_SET(m_seed); \
	\
	scaleModifier *= m_frequency;\
	\
	SIMDf xFreqV = SIMDf_SET(scaleModifier * m_xScale);\
	SIMDf yFreqV = SIMDf_SET(scaleModifier * m_yScale);\
	SIMDf zFreqV = SIMDf_SET(scaleModifier * m_zScale);\
	\
	SET_BUILDER(SIMDf result = FUNC(func##Single)(seedV, xf, yf, zf))\
	\
	SIMD_ZERO_ALL();\
}

#define FILL_FRACTAL_SET(func) \
void SIMD_LEVEL_CLASS::Fill##func##FractalSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)\
{\
	assert(noiseSet);\
	Z_SIZE_ASSERT(zSize);\
	SIMD_ZERO_ALL();\
	\
	SIMDi seedV = SIMDi_SET(m_seed);\
	SIMDf lacunarityV = SIMDf_SET(m_lacunarity);\
	SIMDf gainV = SIMDf_SET(m_gain);\
	\
	scaleModifier *= m_frequency;\
	\
	SIMDf xFreqV = SIMDf_SET(scaleModifier * m_xScale);\
	SIMDf yFreqV = SIMDf_SET(scaleModifier * m_yScale);\
	SIMDf zFreqV = SIMDf_SET(scaleModifier * m_zScale);\
	\
	switch(m_fractalType)\
	{\
	case FBM:\
		SET_BUILDER(FBM_SINGLE(func))\
		break;\
	case Billow:\
		SET_BUILDER(BILLOW_SINGLE(func))\
		break;\
	case RigidMulti:\
		SET_BUILDER(RIGIDMULTI_SINGLE(func))\
		break;\
	}\
	SIMD_ZERO_ALL();\
}

FILL_SET(Value)
FILL_FRACTAL_SET(Value)

FILL_SET(Gradient)
FILL_FRACTAL_SET(Gradient)

FILL_SET(Simplex)
FILL_FRACTAL_SET(Simplex)

#ifdef FN_ALIGNED_SETS
#define SIZE_MASK
#define SAFE_LAST(f)
#else
#define SIZE_MASK & ~(VECTOR_SIZE - 1)
#define SAFE_LAST(f)\
if (loopMax != vectorSet->size)\
{\
	std::size_t remaining = (vectorSet->size - loopMax) * 4;\
	\
	SIMDf xf = SIMDf_SET_ZERO();\
	SIMDf yf = SIMDf_SET_ZERO();\
	SIMDf zf = SIMDf_SET_ZERO();\
	\
	std::memcpy(&xf, &vectorSet->xSet[loopMax], remaining);\
	std::memcpy(&yf, &vectorSet->ySet[loopMax], remaining);\
	std::memcpy(&zf, &vectorSet->zSet[loopMax], remaining);\
	\
	xf = SIMDf_MUL_ADD(xf, freqV, xOffsetV);\
	yf = SIMDf_MUL_ADD(yf, freqV, yOffsetV);\
	zf = SIMDf_MUL_ADD(zf, freqV, zOffsetV);\
	\
	f;\
	std::memcpy(&noiseSet[index], &result, remaining);\
}
#endif

#define VECTOR_SET_BUILDER(f)\
while (index < loopMax)\
{\
	SIMDf xf = SIMDf_MUL_ADD(SIMDf_LOAD(&vectorSet->xSet[index]), freqV, xOffsetV);\
	SIMDf yf = SIMDf_MUL_ADD(SIMDf_LOAD(&vectorSet->ySet[index]), freqV, yOffsetV);\
	SIMDf zf = SIMDf_MUL_ADD(SIMDf_LOAD(&vectorSet->zSet[index]), freqV, zOffsetV);\
	\
	f;\
	SIMDf_STORE(&noiseSet[index], result);\
	index += VECTOR_SIZE;\
}\
SAFE_LAST(f)

#define FILL_VECTOR_SET(func)\
void SIMD_LEVEL_CLASS::Fill##func##Set(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)\
{\
	assert(noiseSet);\
	assert(vectorSet);\
	assert(vectorSet->size >= 0);\
	SIMD_ZERO_ALL();\
	\
	SIMDi seedV = SIMDi_SET(m_seed);\
	SIMDf freqV = SIMDf_SET(m_frequency);\
	SIMDf xOffsetV = SIMDf_SET(xOffset*m_frequency);\
	SIMDf yOffsetV = SIMDf_SET(yOffset*m_frequency);\
	SIMDf zOffsetV = SIMDf_SET(zOffset*m_frequency);\
	\
	int index = 0;\
	int loopMax = vectorSet->size SIZE_MASK;\
	\
	VECTOR_SET_BUILDER(SIMDf result = FUNC(func##Single)(seedV, xf, yf, zf))\
	SIMD_ZERO_ALL();\
}

#define FILL_FRACTAL_VECTOR_SET(func)\
void SIMD_LEVEL_CLASS::Fill##func##FractalSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)\
{\
	assert(noiseSet);\
	assert(vectorSet);\
	assert(vectorSet->size >= 0);\
	SIMD_ZERO_ALL();\
	\
	SIMDi seedV = SIMDi_SET(m_seed);\
	SIMDf freqV = SIMDf_SET(m_frequency);\
	SIMDf lacunarityV = SIMDf_SET(m_lacunarity);\
	SIMDf gainV = SIMDf_SET(m_gain);\
	SIMDf xOffsetV = SIMDf_SET(xOffset*m_frequency);\
	SIMDf yOffsetV = SIMDf_SET(yOffset*m_frequency);\
	SIMDf zOffsetV = SIMDf_SET(zOffset*m_frequency);\
	\
	int index = 0;\
	int loopMax = vectorSet->size SIZE_MASK;\
	\
	switch(m_fractalType)\
	{\
	case FBM:\
		VECTOR_SET_BUILDER(FBM_SINGLE(func))\
		break;\
	case Billow:\
		VECTOR_SET_BUILDER(BILLOW_SINGLE(func))\
		break;\
	case RigidMulti:\
		VECTOR_SET_BUILDER(RIGIDMULTI_SINGLE(func))\
		break;\
	}\
	SIMD_ZERO_ALL();\
}

	FILL_VECTOR_SET(Value)
	FILL_FRACTAL_VECTOR_SET(Value)

	FILL_VECTOR_SET(Gradient)
	FILL_FRACTAL_VECTOR_SET(Gradient)

	FILL_VECTOR_SET(Simplex)
	FILL_FRACTAL_VECTOR_SET(Simplex)

	FILL_VECTOR_SET(WhiteNoise)

	void SIMD_LEVEL_CLASS::FillWhiteNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
{
	assert(noiseSet);
	Z_SIZE_ASSERT(zSize);
	SIMD_ZERO_ALL();
	SIMDi seedV = SIMDi_SET(m_seed);

	if ((zSize & (VECTOR_SIZE - 1)) == 0)
	{
		SIMDi yBase = SIMDi_SET(yStart);
		SIMDi zBase = SIMDi_ADD(SIMDi_NUM(incremental), SIMDi_SET(zStart));

		SIMDi x = SIMDi_SET(xStart);

		int index = 0;

		for (int ix = 0; ix < xSize; ix++)
		{
			SIMDi y = yBase;

			for (int iy = 0; iy < ySize; iy++)
			{
				SIMDi z = zBase;

				SIMDf_STORE(&noiseSet[index], FUNC(ValCoord)(seedV, x, y, z));

				int iz = VECTOR_SIZE;
				while (iz < zSize)
				{
					z = SIMDi_ADD(z, SIMDi_NUM(vectorSize));
					index += VECTOR_SIZE;
					iz += VECTOR_SIZE;

					SIMDf_STORE(&noiseSet[index], FUNC(ValCoord)(seedV, x, y, z));
				}
				index += VECTOR_SIZE;
				y = SIMDi_ADD(y, SIMDi_NUM(1));
			}
			x = SIMDi_ADD(x, SIMDi_NUM(1));
		}
	}
	else
	{

		SIMDi ySizeV = SIMDi_SET(ySize);
		SIMDi zSizeV = SIMDi_SET(zSize);

		SIMDi yEndV = SIMDi_SET(yStart + ySize - 1);
		SIMDi zEndV = SIMDi_SET(zStart + zSize - 1);

		SIMDi x = SIMDi_SET(xStart);
		SIMDi y = SIMDi_SET(yStart);
		SIMDi z = SIMDi_ADD(SIMDi_SET(zStart), SIMDi_NUM(incremental));
		AVX_DOUBLE_RESET;

		int index = 0;
		int maxIndex = xSize * ySize * zSize;

		for (; index < maxIndex - VECTOR_SIZE; index += VECTOR_SIZE)
		{
			SIMDf_STORE(&noiseSet[index], FUNC(ValCoord)(seedV, x, y, z));

			z = SIMDi_ADD(z, SIMDi_NUM(vectorSize));

			SIMDi zReset = SIMDi_GREATER_THAN(z, zEndV);
			y = SIMDi_ADD(y, SIMDi_AND(SIMDi_NUM(1), zReset));
			z = SIMDi_SUB(z, SIMDi_AND(zSizeV, zReset));

			SIMDi yReset = SIMDi_GREATER_THAN(y, yEndV);
			x = SIMDi_ADD(x, SIMDi_AND(SIMDi_NUM(1), yReset));
			y = SIMDi_SUB(y, SIMDi_AND(ySizeV, yReset));

			AVX_DOUBLE_RESET;
		}
		SIMDf result = FUNC(ValCoord)(seedV, x, y, z);
		STORE_LAST_RESULT(&noiseSet[index], result);
	}
	SIMD_ZERO_ALL();
}

#define CELLULAR_MULTI(returnFunc)\
switch(m_cellularDistanceFunction)\
{\
case Euclidean:\
	SET_BUILDER(SIMDf result = FUNC(Cellular##returnFunc##EuclideanSingle)(seedV, xf, yf, zf))\
	break;\
case Manhattan:\
	SET_BUILDER(SIMDf result = FUNC(Cellular##returnFunc##ManhattanSingle)(seedV, xf, yf, zf))\
	break;\
case Natural:\
	SET_BUILDER(SIMDf result = FUNC(Cellular##returnFunc##NaturalSingle)(seedV, xf, yf, zf))\
	break;\
}

void SIMD_LEVEL_CLASS::FillCellularSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
{
	assert(noiseSet);
	Z_SIZE_ASSERT(zSize);
	SIMD_ZERO_ALL();
	SIMDi seedV = SIMDi_SET(m_seed);

	scaleModifier *= m_frequency;

	SIMDf xFreqV = SIMDf_SET(scaleModifier * m_xScale);
	SIMDf yFreqV = SIMDf_SET(scaleModifier * m_yScale);
	SIMDf zFreqV = SIMDf_SET(scaleModifier * m_zScale);

	switch (m_cellularReturnType)
	{
	case CellValue:
		CELLULAR_MULTI(Value);
		break;
	case Distance:
		CELLULAR_MULTI(Distance);
		break;
	case Distance2:
		CELLULAR_MULTI(Distance2);
		break;
	case Distance2Add:
		CELLULAR_MULTI(Distance2Add);
		break;
	case Distance2Sub:
		CELLULAR_MULTI(Distance2Sub);
		break;
	case Distance2Mul:
		CELLULAR_MULTI(Distance2Mul);
		break;
	case Distance2Div:
		CELLULAR_MULTI(Distance2Div);
		break;
	}
	SIMD_ZERO_ALL();
}

#define CELLULAR_MULTI_VECTOR(returnFunc)\
switch(m_cellularDistanceFunction)\
{\
case Euclidean:\
	VECTOR_SET_BUILDER(SIMDf result = FUNC(Cellular##returnFunc##EuclideanSingle)(seedV, xf, yf, zf))\
	break;\
case Manhattan:\
	VECTOR_SET_BUILDER(SIMDf result = FUNC(Cellular##returnFunc##ManhattanSingle)(seedV, xf, yf, zf))\
	break;\
case Natural:\
	VECTOR_SET_BUILDER(SIMDf result = FUNC(Cellular##returnFunc##NaturalSingle)(seedV, xf, yf, zf))\
	break;\
}

void SIMD_LEVEL_CLASS::FillCellularSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)
{
	assert(noiseSet);
	assert(vectorSet);
	assert(vectorSet->size >= 0);
	SIMD_ZERO_ALL();

	SIMDi seedV = SIMDi_SET(m_seed);
	SIMDf freqV = SIMDf_SET(m_frequency);
	SIMDf xOffsetV = SIMDf_SET(xOffset*m_frequency);
	SIMDf yOffsetV = SIMDf_SET(yOffset*m_frequency);
	SIMDf zOffsetV = SIMDf_SET(zOffset*m_frequency);

	int index = 0;
	int loopMax = vectorSet->size SIZE_MASK;

	switch (m_cellularReturnType)
	{
	case CellValue:
		CELLULAR_MULTI_VECTOR(Value);
		break;
	case Distance:
		CELLULAR_MULTI_VECTOR(Distance);
		break;
	case Distance2:
		CELLULAR_MULTI_VECTOR(Distance2);
		break;
	case Distance2Add:
		CELLULAR_MULTI_VECTOR(Distance2Add);
		break;
	case Distance2Sub:
		CELLULAR_MULTI_VECTOR(Distance2Sub);
		break;
	case Distance2Mul:
		CELLULAR_MULTI_VECTOR(Distance2Mul);
		break;
	case Distance2Div:
		CELLULAR_MULTI_VECTOR(Distance2Div);
		break;
	}
	SIMD_ZERO_ALL();
}

#define SAMPLE_INDEX(_x,_y,_z) ((_x) * yzSizeSample + (_y) * zSizeSample + (_z))
#define SET_INDEX(_x,_y,_z) ((_x) * yzSize + (_y) * zSize + (_z))

void SIMD_LEVEL_CLASS::FillSampledNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, int sampleScale)
{
	assert(noiseSet);
	SIMD_ZERO_ALL();

	if (sampleScale <= 0)
	{
		FillNoiseSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize);
		return;
	}

	int sampleSize = 1 << sampleScale;
	int sampleMask = sampleSize - 1;
	float scaleModifier = float(sampleSize);

	int xOffset = (sampleSize - (xStart & sampleMask)) & sampleMask;
	int yOffset = (sampleSize - (yStart & sampleMask)) & sampleMask;
	int zOffset = (sampleSize - (zStart & sampleMask)) & sampleMask;

	int xSizeSample = xSize + xOffset;
	int ySizeSample = ySize + yOffset;
	int zSizeSample = zSize + zOffset;

	if (xSizeSample & sampleMask)
		xSizeSample = (xSizeSample & ~sampleMask) + sampleSize;

	if (ySizeSample & sampleMask)
		ySizeSample = (ySizeSample & ~sampleMask) + sampleSize;

	if (zSizeSample & sampleMask)
		zSizeSample = (zSizeSample & ~sampleMask) + sampleSize;

	xSizeSample = (xSizeSample >> sampleScale) + 1;
	ySizeSample = (ySizeSample >> sampleScale) + 1;
	zSizeSample = (zSizeSample >> sampleScale) + 1;

	Z_SIZE_ASSERT(zSizeSample);

	float* noiseSetSample = GetEmptySet(xSizeSample * ySizeSample * zSizeSample);
	FillNoiseSet(noiseSetSample, xStart >> sampleScale, yStart >> sampleScale, zStart >> sampleScale, xSizeSample, ySizeSample, zSizeSample, scaleModifier);

	int yzSizeSample = ySizeSample * zSizeSample;
	int yzSize = ySize * zSize;

	SIMDi axisMask = SIMDi_SET(sampleMask);
	SIMDf axisScale = SIMDf_SET(1.f / scaleModifier);
	SIMDf axisOffset = SIMDf_MUL(axisScale, SIMDf_NUM(0_5));

	SIMDi sampleSizeSIMD = SIMDi_SET(sampleSize);
	SIMDi xSIMD = SIMDi_SET(-xOffset);
	SIMDi yBase = SIMDi_SET(-yOffset);
	SIMDi zBase = SIMDi_SET(-zOffset);

	for (int x = 0; x < xSizeSample - 1; x++)
	{
		SIMDi ySIMD = yBase;
		for (int y = 0; y < ySizeSample - 1; y++)
		{
			SIMDi zSIMD = zBase;

			SIMDf c001 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y, 0)]);
			SIMDf c101 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y, 0)]);
			SIMDf c011 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y + 1, 0)]);
			SIMDf c111 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, 0)]);
			for (int z = 0; z < zSizeSample - 1; z++)
			{
				SIMDf c000 = c001;
				SIMDf c100 = c101;
				SIMDf c010 = c011;
				SIMDf c110 = c111;
				c001 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y, z + 1)]);
				c101 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y, z + 1)]);
				c011 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y + 1, z + 1)]);
				c111 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, z + 1)]);

				SIMDi localCountSIMD = SIMDi_NUM(incremental);

				int localCount = 0;
				while (localCount < (1 << (sampleScale * 3)))
				{
					uSIMDi xi, yi, zi;
					xi.m = SIMDi_AND(SIMDi_SHIFT_R(localCountSIMD, sampleScale * 2), axisMask);
					yi.m = SIMDi_AND(SIMDi_SHIFT_R(localCountSIMD, sampleScale), axisMask);
					zi.m = SIMDi_AND(localCountSIMD, axisMask);

					SIMDf xf = SIMDf_MUL_ADD(SIMDf_CONVERT_TO_FLOAT(xi.m), axisScale, axisOffset);
					SIMDf yf = SIMDf_MUL_ADD(SIMDf_CONVERT_TO_FLOAT(yi.m), axisScale, axisOffset);
					SIMDf zf = SIMDf_MUL_ADD(SIMDf_CONVERT_TO_FLOAT(zi.m), axisScale, axisOffset);

					xi.m = SIMDi_ADD(xi.m, xSIMD);
					yi.m = SIMDi_ADD(yi.m, ySIMD);
					zi.m = SIMDi_ADD(zi.m, zSIMD);

					uSIMDf sampledResults;
					sampledResults.m = FUNC(Lerp)(
						FUNC(Lerp)(
							FUNC(Lerp)(c000, c100, xf),
							FUNC(Lerp)(c010, c110, xf), yf),
						FUNC(Lerp)(
							FUNC(Lerp)(c001, c101, xf),
							FUNC(Lerp)(c011, c111, xf), yf), zf);

					for (int i = 0; i < VECTOR_SIZE; i++)
					{
						if (xi.a[i] >= 0 && xi.a[i] < xSize &&
							yi.a[i] >= 0 && yi.a[i] < ySize &&
							zi.a[i] >= 0 && zi.a[i] < zSize)
						{
							int index = SET_INDEX(xi.a[i], yi.a[i], zi.a[i]);
							noiseSet[index] = sampledResults.a[i];
						}
					}

					localCount += VECTOR_SIZE;
					localCountSIMD = SIMDi_ADD(localCountSIMD, SIMDi_NUM(vectorSize));
				}
				zSIMD = SIMDi_ADD(zSIMD, sampleSizeSIMD);
			}
			ySIMD = SIMDi_ADD(ySIMD, sampleSizeSIMD);
		}
		xSIMD = SIMDi_ADD(xSIMD, sampleSizeSIMD);
	}

	FreeNoiseSet(noiseSetSample);
	SIMD_ZERO_ALL();
}

void SIMD_LEVEL_CLASS::FillSampledNoiseSet(float* noiseSet, FastNoiseVectorSet* vectorSet, int xSize, int ySize, int zSize, float xOffset, float yOffset, float zOffset)
{
	assert(noiseSet);
	assert(vectorSet);
	assert(vectorSet->size >= 0);
	SIMD_ZERO_ALL();

	int sampleScale = vectorSet->sampleScale;

	if (sampleScale <= 0)
	{
		FillNoiseSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
		return;
	}

	int sampleSize = 1 << sampleScale;
	int sampleMask = sampleSize - 1;
	float scaleModifier = float(sampleSize);

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

	float* noiseSetSample = GetEmptySet(vectorSet->size);
	FillNoiseSet(noiseSetSample, vectorSet, xOffset, yOffset, zOffset);

	int yzSizeSample = ySizeSample * zSizeSample;
	int yzSize = ySize * zSize;

	SIMDi axisMask = SIMDi_SET(sampleMask);
	SIMDf axisScale = SIMDf_SET(1.f / scaleModifier);
	SIMDf axisOffset = SIMDf_MUL(axisScale, SIMDf_NUM(0_5));

	SIMDi sampleSizeSIMD = SIMDi_SET(sampleSize);
	SIMDi xSIMD = SIMDi_SET_ZERO();

	for (int x = 0; x < xSizeSample - 1; x++)
	{
		SIMDi ySIMD = SIMDi_SET_ZERO();
		for (int y = 0; y < ySizeSample - 1; y++)
		{
			SIMDi zSIMD = SIMDi_SET_ZERO();

			SIMDf c001 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y, 0)]);
			SIMDf c101 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y, 0)]);
			SIMDf c011 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y + 1, 0)]);
			SIMDf c111 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, 0)]);
			for (int z = 0; z < zSizeSample - 1; z++)
			{
				SIMDf c000 = c001;
				SIMDf c100 = c101;
				SIMDf c010 = c011;
				SIMDf c110 = c111;
				c001 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y, z + 1)]);
				c101 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y, z + 1)]);
				c011 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x, y + 1, z + 1)]);
				c111 = SIMDf_SET(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, z + 1)]);

				SIMDi localCountSIMD = SIMDi_NUM(incremental);

				int localCount = 0;
				while (localCount < (1 << (sampleScale * 3)))
				{
					uSIMDi xi, yi, zi;
					xi.m = SIMDi_AND(SIMDi_SHIFT_R(localCountSIMD, sampleScale * 2), axisMask);
					yi.m = SIMDi_AND(SIMDi_SHIFT_R(localCountSIMD, sampleScale), axisMask);
					zi.m = SIMDi_AND(localCountSIMD, axisMask);

					SIMDf xf = SIMDf_MUL_ADD(SIMDf_CONVERT_TO_FLOAT(xi.m), axisScale, axisOffset);
					SIMDf yf = SIMDf_MUL_ADD(SIMDf_CONVERT_TO_FLOAT(yi.m), axisScale, axisOffset);
					SIMDf zf = SIMDf_MUL_ADD(SIMDf_CONVERT_TO_FLOAT(zi.m), axisScale, axisOffset);

					xi.m = SIMDi_ADD(xi.m, xSIMD);
					yi.m = SIMDi_ADD(yi.m, ySIMD);
					zi.m = SIMDi_ADD(zi.m, zSIMD);

					uSIMDf sampledResults;
					sampledResults.m = FUNC(Lerp)(
						FUNC(Lerp)(
							FUNC(Lerp)(c000, c100, xf),
							FUNC(Lerp)(c010, c110, xf), yf),
						FUNC(Lerp)(
							FUNC(Lerp)(c001, c101, xf),
							FUNC(Lerp)(c011, c111, xf), yf), zf);

					for (int i = 0; i < VECTOR_SIZE; i++)
					{
						if (xi.a[i] < xSize &&
							yi.a[i] < ySize &&
							zi.a[i] < zSize)
						{
							int index = SET_INDEX(xi.a[i], yi.a[i], zi.a[i]);
							noiseSet[index] = sampledResults.a[i];
						}
					}

					localCount += VECTOR_SIZE;
					localCountSIMD = SIMDi_ADD(localCountSIMD, SIMDi_NUM(vectorSize));
				}
				zSIMD = SIMDi_ADD(zSIMD, sampleSizeSIMD);
			}
			ySIMD = SIMDi_ADD(ySIMD, sampleSizeSIMD);
		}
		xSIMD = SIMDi_ADD(xSIMD, sampleSizeSIMD);
	}

	FreeNoiseSet(noiseSetSample);
	SIMD_ZERO_ALL();
}

#undef SIMD_LEVEL
#endif
