#ifndef SIMD_LEVEL
#include "FastNoiseSIMD.h"
#endif

#if defined(SIMD_LEVEL) || defined(COMPILE_AVX2)

#ifndef SIMD_LEVEL
#define SIMD_LEVEL AVX2
#include <immintrin.h> 
#ifndef __AVX__
#error To compile AVX2 set C++ code generation to use /arch:AVX on FastNoiseSIMD_internal.cpp, or remove "#define COMPILE_AVX2" from FastNoiseSIMD.h
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
#if SIMD_LEVEL == AVX2
#define VECTOR_SIZE 8
#define MEMORY_ALIGNMENT 32
typedef __m256 SIMDf;
typedef __m256i SIMDi;
#define SIMDf_SET(a) _mm256_set1_ps(a)
#define SIMDf_SET_ZERO() _mm256_setzero_ps()
#define SIMDi_SET(a) _mm256_set1_epi32(a)
#define SIMDi_SET_ZERO() _mm256_setzero_si256()

#elif SIMD_LEVEL >= SSE2 
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
#if SIMD_LEVEL > NO_SIMD_FALLBACK
#define SIMD_ALIGNED_SET(floatCount) (float*)_aligned_malloc(floatCount* sizeof(float), MEMORY_ALIGNMENT)
#else
#define SIMD_ALIGNED_SET(floatCount) new float[floatCount]
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

#define SIMDf_NUM_DEFINE_NAME(n,x) const static SIMDf SIMDf_NUM(n) = SIMDf_SET(x)
#define SIMDf_NUM_DEFINE(x) SIMDf_NUM_DEFINE_NAME(x,x)
#define SIMDi_NUM_DEFINE_NAME(n,x) const static SIMDi SIMDi_NUM(n) = SIMDi_SET(x)
#define SIMDi_NUM_DEFINE(x) SIMDi_NUM_DEFINE_NAME(x,x)

SIMDf_NUM_DEFINE(0);
SIMDf_NUM_DEFINE(1);
SIMDf_NUM_DEFINE(6);
SIMDf_NUM_DEFINE(10);
SIMDf_NUM_DEFINE(15);

SIMDi_NUM_DEFINE(0);
SIMDi_NUM_DEFINE(1);
SIMDi_NUM_DEFINE(2);
SIMDi_NUM_DEFINE(4);
SIMDi_NUM_DEFINE(8);
SIMDi_NUM_DEFINE(12);
SIMDi_NUM_DEFINE(14);
SIMDi_NUM_DEFINE(15);
SIMDi_NUM_DEFINE(60493);
SIMDi_NUM_DEFINE(19990303);

SIMDi_NUM_DEFINE_NAME(xPrime, 1619);
SIMDi_NUM_DEFINE_NAME(yPrime, 31337);
SIMDi_NUM_DEFINE_NAME(zPrime, 6971);

// SIMD functions
#if SIMD_LEVEL >= AVX2

#define SIMDf_STORE(p,a) _mm256_stream_ps(p,a)
#define SIMDf_LOAD(p) _mm256_load_ps(p)

#define SIMDf_ADD(a,b) _mm256_add_ps(a,b)
#define SIMDf_SUB(a,b) _mm256_sub_ps(a,b)
#define SIMDf_MUL(a,b) _mm256_mul_ps(a,b)
#define SIMDf_DIV(a,b) _mm256_div_ps(a,b)

#define SIMDf_FLOOR(a) _mm256_floor_ps(a)
#define SIMDf_BLENDV(a,b,mask) _mm256_blendv_ps(a,b,mask)
#define SIMDf_GATHER(p,a) _mm256_i32gather_ps(p,a,4)

#define SIMDi_ADD(a,b) _mm256_add_epi32(a,b)
#define SIMDi_SUB(a,b) _mm256_sub_epi32(a,b)
#define SIMDi_MUL(a,b) _mm256_mullo_epi32(a,b)

#define SIMDi_AND(a,b) _mm256_and_si256(a,b)
#define SIMDi_OR(a,b) _mm256_or_si256(a,b)
#define SIMDi_XOR(a,b) _mm256_xor_si256(a,b)
#define SIMDi_SHIFT_R(a, b) _mm256_srli_epi32(a, b)
#define SIMDi_SHIFT_L(a, b) _mm256_slli_epi32(a, b)

#define SIMDi_EQUAL(a,b) _mm256_cmpeq_epi32(a,b)
#define SIMDi_GREATER_THAN(a,b) _mm256_cmpgt_epi32(a,b)
#define SIMDi_LESS_THAN(a,b) _mm256_cmpgt_epi32(b,a)

#define SIMDf_CONVERT_TO_FLOAT(a) _mm256_cvtepi32_ps(a)
#define SIMDf_CAST_TO_FLOAT(a) _mm256_castsi256_ps(a)
#define SIMDi_CONVERT_TO_INT(a) _mm256_cvtps_epi32(a)
#define SIMDi_CAST_TO_INT(a) _mm256_castps_si256(a)

#elif SIMD_LEVEL >= SSE2

#define SIMDf_STORE(p,a) _mm_stream_ps(p,a)
#define SIMDf_LOAD(p) _mm_load_ps(p)

#define SIMDf_ADD(a,b) _mm_add_ps(a,b)
#define SIMDf_SUB(a,b) _mm_sub_ps(a,b)
#define SIMDf_MUL(a,b) _mm_mul_ps(a,b)
#define SIMDf_DIV(a,b) _mm_div_ps(a,b)

#if SIMD_LEVEL == SSE41
#define SIMDi_MUL(a,b) _mm_mullo_epi32(a,b)
#define SIMDf_FLOOR(a) _mm_floor_ps(a)
#define SIMDf_BLENDV(a,b,mask) _mm_blendv_ps(a,b,mask)
#else
inline static SIMDi FUNC(MUL)(const SIMDi& a, const SIMDi& b)
{
	__m128 tmp1 = _mm_castsi128_ps(_mm_mul_epu32(a, b)); /* mul 2,0*/
	__m128 tmp2 = _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4))); /* mul 3,1 */
	return _mm_castps_si128(_mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(2, 0, 2, 0)));
}
#define SIMDi_MUL(a,b) FUNC(MUL)(a,b)

inline static SIMDf FUNC(FLOOR)(const SIMDf& a)
{
	__m128 fval = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));

	return _mm_sub_ps(fval, _mm_and_ps(_mm_cmplt_ps(a, fval), SIMDf_NUM(1)));
}
#define SIMDf_FLOOR(a) FUNC(FLOOR)(a)

inline static SIMDf FUNC(BLENDV)(const SIMDf& a, const SIMDf& b, const SIMDf& mask)
{
	SIMDf maskInv = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(mask), SIMDi_NUM(0)));

	return _mm_or_ps(_mm_and_ps(maskInv, a), _mm_andnot_ps(maskInv, b));
}

#define SIMDf_BLENDV(a,b,mask) FUNC(BLENDV)(a,b,mask)
#endif

inline static SIMDf FUNC(GATHER)(const float* p, const SIMDi& a)
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

#define SIMDi_AND(a,b) _mm_and_si128(a,b)
#define SIMDi_ADD(a,b) _mm_add_epi32(a,b)
#define SIMDi_SUB(a,b) _mm_sub_epi32(a,b)

#define SIMDi_OR(a,b) _mm_or_si128(a,b)
#define SIMDi_XOR(a,b) _mm_xor_si128(a,b)
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
#define SIMDf_FLOOR(a) floor(a)
#define SIMDf_BLENDV(a,b,mask) (SIMDi_CAST_TO_INT(mask) ? (b) : (a))
#define SIMDf_GATHER(p,a) (*(reinterpret_cast<const float*>(p)+(a)))

#define SIMDi_ADD(a,b) ((a) + (b))
#define SIMDi_SUB(a,b) ((a) - (b))
#define SIMDi_MUL(a,b) ((a) * (b))

#define SIMDi_AND(a,b) ((a) & (b))
#define SIMDi_OR(a,b) ((a) | (b))
#define SIMDi_XOR(a,b) ((a) ^ (b))
#define SIMDi_SHIFT_R(a, b) ((a) >> (b))
#define SIMDi_SHIFT_L(a, b) ((a) << (b))

#define SIMDi_EQUAL(a,b) (((a) == (b)) ? 0xFFFFFFFF : 0)
#define SIMDi_GREATER_THAN(a,b) (((a) > (b)) ? 0xFFFFFFFF : 0)
#define SIMDi_LESS_THAN(a,b) (((a) < (b)) ? 0xFFFFFFFF : 0)

#define SIMDi_CONVERT_TO_INT(a) static_cast<int>(a)
#define SIMDf_CONVERT_TO_FLOAT(a) static_cast<float>(a)
#endif

// FMA2
#if SIMD_LEVEL == AVX2
#define SIMDf_MUL_ADD(a,b,c) _mm256_fmadd_ps(a,b,c) 
#define SIMDf_MUL_SUB(a,b,c) _mm256_fmsub_ps(a,b,c) 
#define SIMD_ZERO_ALL() _mm256_zeroall()
#else
#define SIMDf_MUL_ADD(a,b,c) SIMDf_ADD(SIMDf_MUL(a,b),c)
#define SIMDf_MUL_SUB(a,b,c) SIMDf_SUB(SIMDf_MUL(a,b),c) 
#define SIMD_ZERO_ALL()
#endif

static SIMDf FUNC(GradientSingle)(const SIMDi&, const SIMDf& x, const SIMDf& y, const SIMDf& z);

static bool VAR(SIMD_Values_Set) = false;

static SIMDf SIMDf_NUM(incremental);
static SIMDi SIMDi_NUM(incremental);

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

float* SIMD_LEVEL_CLASS::GetGradientSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance)
{
	SIMD_ZERO_ALL();

	// zSize must be a multiple of VECTOR_SIZE (8)
	if ((zSize & (VECTOR_SIZE - 1)) != 0)
		throw;

	stepDistance *= m_frequency;

	//SIMD data has to be aligned
	float* result = SIMD_ALIGNED_SET(xSize*ySize*zSize);

	SIMDi seed = SIMDi_SET(m_seed);

	SIMDf step = SIMDf_SET(stepDistance);
	SIMDf vStep = SIMDf_SET(stepDistance * VECTOR_SIZE);

	SIMDf yBase = SIMDf_SET(yStart * stepDistance);
	SIMDf zBase = SIMDf_MUL_ADD(step, SIMDf_NUM(incremental), SIMDf_SET(zStart * stepDistance));

	SIMDf x = SIMDf_SET(xStart * stepDistance);
	SIMDf y, z;

	int index = 0;

	for (int ix = 0; ix < xSize; ix++)
	{
		y = yBase;

		for (int iy = 0; iy < ySize; iy++)
		{
			z = zBase;

			for (int iz = 0; iz < zSize; iz += VECTOR_SIZE)
			{
				SIMDf_STORE(&result[index], FUNC(GradientSingle)(seed, x, y, z));

				index += VECTOR_SIZE;

				z = SIMDf_ADD(z, vStep);
			}
			y = SIMDf_ADD(y, step);
		}
		x = SIMDf_ADD(x, step);
	}
	SIMD_ZERO_ALL();

	return result;
}

float* SIMD_LEVEL_CLASS::GetGradientFractalSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance)
{
	SIMD_ZERO_ALL();

	// zSize must be a multiple of VECTOR_SIZE (8)
	if ((zSize & (VECTOR_SIZE - 1)) != 0)
		throw;

	stepDistance *= m_frequency;

	//SIMD data has to be aligned
	float* result = SIMD_ALIGNED_SET(xSize*ySize*zSize);

	SIMDi seed = SIMDi_SET(m_seed);
	SIMDf lacunarity = SIMDf_SET(m_lacunarity);
	SIMDf gain = SIMDf_SET(m_gain);

	SIMDf step = SIMDf_SET(stepDistance);
	SIMDf vStep = SIMDf_SET(stepDistance * VECTOR_SIZE);

	SIMDf yBase = SIMDf_SET(yStart * stepDistance);
	SIMDf zBase = SIMDf_MUL_ADD(step, SIMDf_NUM(incremental), SIMDf_SET(zStart * stepDistance));

	SIMDf x = SIMDf_SET(xStart * stepDistance);
	SIMDf y, z;

	int index = 0;

	for (int ix = 0; ix < xSize; ix++)
	{
		y = yBase;

		for (int iy = 0; iy < ySize; iy++)
		{
			z = zBase;

			for (int iz = 0; iz < zSize; iz += VECTOR_SIZE)
			{
				SIMDi seedF = seed;
				SIMDf sumF = FUNC(GradientSingle)(seed, x, y, z);
				SIMDf maxF = SIMDf_NUM(1);
				SIMDf ampF = SIMDf_NUM(1);
				unsigned int octaveIndex = 0;

				SIMDf xF = x;
				SIMDf yF = y;
				SIMDf zF = z;

				while (++octaveIndex < m_octaves)
				{
					xF = SIMDf_MUL(xF, lacunarity);
					yF = SIMDf_MUL(yF, lacunarity);
					zF = SIMDf_MUL(zF, lacunarity);
					seedF = SIMDi_ADD(seedF, SIMDi_NUM(1));

					ampF = SIMDf_MUL(ampF, gain);
					maxF = SIMDf_ADD(maxF, ampF);
					sumF = SIMDf_MUL_ADD(FUNC(GradientSingle)(seedF, xF, yF, zF), ampF, sumF);
				}

				SIMDf_STORE(&result[index], SIMDf_DIV(sumF, maxF));

				index += VECTOR_SIZE;

				z = SIMDf_ADD(z, vStep);
			}
			y = SIMDf_ADD(y, step);
		}
		x = SIMDf_ADD(x, step);
	}
	SIMD_ZERO_ALL();

	return result;
}

static SIMDi FUNC(Hash)(const SIMDi& seed, const SIMDi& x, const SIMDi& y, const SIMDi& z)
{
	SIMDi hash;

	hash = SIMDi_MUL(x, SIMDi_NUM(xPrime));
	hash = SIMDi_ADD(SIMDi_MUL(y, SIMDi_NUM(yPrime)), hash);
	hash = SIMDi_ADD(SIMDi_MUL(z, SIMDi_NUM(zPrime)), hash);
	hash = SIMDi_XOR(hash, seed);

	hash = SIMDi_MUL(SIMDi_ADD(SIMDi_MUL(SIMDi_MUL(hash, hash), SIMDi_NUM(60493)), SIMDi_NUM(19990303)), hash);
	hash = SIMDi_XOR(SIMDi_SHIFT_R(hash, 13), hash);

	return hash;
}

static SIMDf FUNC(GradCoord)(const SIMDf& x, const SIMDf& y, const SIMDf& z, SIMDi hash)
{
	hash = SIMDi_AND(hash, SIMDi_NUM(15));

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

	SIMDf xl00 = FUNC(Lerp)(FUNC(GradCoord)(xf0, yf0, zf0, FUNC(Hash)(seed, x0, y0, z0)), FUNC(GradCoord)(xf1, yf0, zf0, FUNC(Hash)(seed, x1, y0, z0)), xs);
	SIMDf xl10 = FUNC(Lerp)(FUNC(GradCoord)(xf0, yf1, zf0, FUNC(Hash)(seed, x0, y1, z0)), FUNC(GradCoord)(xf1, yf1, zf0, FUNC(Hash)(seed, x1, y1, z0)), xs);
	SIMDf xl01 = FUNC(Lerp)(FUNC(GradCoord)(xf0, yf0, zf1, FUNC(Hash)(seed, x0, y0, z1)), FUNC(GradCoord)(xf1, yf0, zf1, FUNC(Hash)(seed, x1, y0, z1)), xs);
	SIMDf xl11 = FUNC(Lerp)(FUNC(GradCoord)(xf0, yf1, zf1, FUNC(Hash)(seed, x0, y1, z1)), FUNC(GradCoord)(xf1, yf1, zf1, FUNC(Hash)(seed, x1, y1, z1)), xs);

	SIMDf yl0 = FUNC(Lerp)(xl00, xl10, ys);
	SIMDf yl1 = FUNC(Lerp)(xl01, xl11, ys);

	return FUNC(Lerp)(yl0, yl1, zs);
}
#undef SIMD_LEVEL
#endif
