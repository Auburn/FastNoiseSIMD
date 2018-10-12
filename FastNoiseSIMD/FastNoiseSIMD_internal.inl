// FastNoiseSIMD_internal.cpp
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

//#include "FastNoiseSIMD.h"
#include <assert.h> 

#ifndef _WIN32
#include <cstdlib>
#endif

// Memory Allocation
#if SIMD_LEVEL > FN_NO_SIMD_FALLBACK && defined(FN_ALIGNED_SETS)
#ifdef _WIN32
#define SIMD_ALLOCATE_SET(floatP, floatCount) floatP = (float*)_aligned_malloc((floatCount)* sizeof(float), MEMORY_ALIGNMENT)
#else
#include <cstdlib>
#define SIMD_ALLOCATE_SET(floatP, floatCount) posix_memalign((void**)&floatP, MEMORY_ALIGNMENT, (floatCount)* sizeof(float))
#endif
#else
#define SIMD_ALLOCATE_SET(floatP, floatCount) floatP = new float[floatCount]
#endif

namespace FastNoise
{
namespace details
{

template<SIMDType _SIMDType>
const bool NoiseSIMD<_SIMDType>::m_registered=FastNoise::NoiseSIMD::registerNoiseSimd(_SIMDType, NoiseSIMD<_SIMDType>::create, NoiseSIMD<_SIMDType>::AlignedSize, NoiseSIMD<_SIMDType>::GetEmptySet);

//template<SIMDType _SIMDType>
//size_t alignedSize(size_t size)
//{
//#ifdef FN_ALIGNED_SETS
//    // size must be a multiple of SIMD<_SIMDType>::vectorSize() (8)
//    if((size & (SIMD<_SIMDType>::vectorSize()-1))!=0)
//    {
//        size&=~(SIMD<_SIMDType>::vectorSize()-1);
//        size+=SIMD<_SIMDType>::vectorSize();
//    }
//#endif
//    return size;
//}

template<SIMDType _SIMDType>
struct simdAlloc
{
    static float *_(size_t count)
    {
#ifdef FN_ALIGNED_SETS
#   ifdef _WIN32
        return (float*)_aligned_malloc((count)*sizeof(float), SIMD<_SIMDType>::alignment());
#   else
        float *ptr;

        posix_memalign((void**)&ptr, SIMD<_SIMDType>::alignment(), (count)*sizeof(float));
        return ptr;
#   endif
#else
        return new float[count];
#endif
    }
};

template<>
struct simdAlloc<SIMDType::None>
{
    static float *_(size_t count) { return new float[count]; }
};

//template<SIMDType _SIMDType>
//float *GetEmptySet(size_t size)
//{
//    size_t aSize=alignedSize<_SIMDType>(size);
//    float* noiseSet=simdAlloc<_SIMDType>::_(aSize);
//
//    return noiseSet;
//}

template<SIMDType _SIMDType>
static typename SIMD<_SIMDType>::Float VECTORCALL Lerp(typename SIMD<_SIMDType>::Float a, typename SIMD<_SIMDType>::Float b, typename SIMD<_SIMDType>::Float t)
{
    typename SIMD<_SIMDType>::Float r;
    r=SIMD<_SIMDType>::sub(b, a);
    r=SIMD<_SIMDType>::mulAdd(r, t, a);
    return r;
}

template<SIMDType _SIMDType>
static typename SIMD<_SIMDType>::Float VECTORCALL InterpQuintic(typename SIMD<_SIMDType>::Float t)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

    typename SIMD<_SIMDType>::Float r;
    r=SIMD<_SIMDType>::mulSub(t, Constant::numf_6, Constant::numf_15);
    r=SIMD<_SIMDType>::mulAdd(r, t, Constant::numf_10);
    r=SIMD<_SIMDType>::mulf(r, t);
    r=SIMD<_SIMDType>::mulf(r, t);
    r=SIMD<_SIMDType>::mulf(r, t);

    return r;
}

template<SIMDType _SIMDType>
static typename SIMD<_SIMDType>::Float VECTORCALL CubicLerp(typename SIMD<_SIMDType>::Float a, typename SIMD<_SIMDType>::Float b, typename SIMD<_SIMDType>::Float c, typename SIMD<_SIMDType>::Float d, typename SIMD<_SIMDType>::Float t)
{
    typename SIMD<_SIMDType>::Float p=SIMD<_SIMDType>::sub(SIMD<_SIMDType>::sub(d, c), SIMD<_SIMDType>::sub(a, b));
    return SIMD<_SIMDType>::mulAdd(t, SIMD<_SIMDType>::mulf(t, SIMD<_SIMDType>::mulf(t, p)), SIMD<_SIMDType>::mulAdd(t, SIMD<_SIMDType>::mulf(t, SIMD<_SIMDType>::sub(SIMD<_SIMDType>::sub(a, b), p)), SIMD<_SIMDType>::mulAdd(t, SIMD<_SIMDType>::sub(c, a), b)));
}

//static typename SIMD<_SIMDType>::Float VECTORCALL FUNC(InterpHermite)(typename SIMD<_SIMDType>::Float t)
//{
//	typename SIMD<_SIMDType>::Float r;
//	r = SIMD<_SIMDType>::mul(t, Constants::numf_2);
//	r = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::add(Constants::numf_1, Constants::numf_2), r);
//	r = SIMD<_SIMDType>::mul(r, t);
//	r = SIMD<_SIMDType>::mul(r, t);
//
//	return r;
//}

template<SIMDType _SIMDType>
static typename SIMD<_SIMDType>::Int VECTORCALL Hash(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Int x, typename SIMD<_SIMDType>::Int y, typename SIMD<_SIMDType>::Int z)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
    typename SIMD<_SIMDType>::Int hash=seed;

    hash=SIMD<_SIMDType>::_xor(x, hash);
    hash=SIMD<_SIMDType>::_xor(y, hash);
    hash=SIMD<_SIMDType>::_xor(z, hash);

    hash=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::mul(SIMD<_SIMDType>::mul(hash, hash), Constant::numi_60493), hash);
    hash=SIMD<_SIMDType>::_xor(SIMD<_SIMDType>::shiftR(hash, 13), hash);

    return hash;
}

template<SIMDType _SIMDType>
static typename SIMD<_SIMDType>::Int VECTORCALL HashHB(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Int x, typename SIMD<_SIMDType>::Int y, typename SIMD<_SIMDType>::Int z)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
    typename SIMD<_SIMDType>::Int hash=seed;

    hash=SIMD<_SIMDType>::_xor(x, hash);
    hash=SIMD<_SIMDType>::_xor(y, hash);
    hash=SIMD<_SIMDType>::_xor(z, hash);
    //hash = SIMD<_SIMDType>::_xor(SIMD<_SIMDType>::shiftR(hash, 13), hash);

    hash=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::mul(SIMD<_SIMDType>::mul(hash, hash), Constant::numi_60493), hash);

    return hash;
}

template<SIMDType _SIMDType>
static typename SIMD<_SIMDType>::Float VECTORCALL ValCoord(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Int x, typename SIMD<_SIMDType>::Int y, typename SIMD<_SIMDType>::Int z)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
    // High bit hash
    typename SIMD<_SIMDType>::Int hash=seed;

    hash=SIMD<_SIMDType>::_xor(x, hash);
    hash=SIMD<_SIMDType>::_xor(y, hash);
    hash=SIMD<_SIMDType>::_xor(z, hash);

    hash=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::mul(SIMD<_SIMDType>::mul(hash, hash), Constant::numi_60493), hash);
    //hash = SIMD<_SIMDType>::_xor(SIMD<_SIMDType>::shiftL(hash, 13), hash);

    return SIMD<_SIMDType>::mulf(Constant::numf_hash2Float, SIMD<_SIMDType>::convert(hash));
}

template<SIMDType _SIMDType>
struct GradCoord
{
    static typename SIMD<_SIMDType>::Float VECTORCALL _(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Int xi, typename SIMD<_SIMDType>::Int yi, typename SIMD<_SIMDType>::Int zi, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
        typename SIMD<_SIMDType>::Int hash=Hash<_SIMDType>(seed, xi, yi, zi);
        typename SIMD<_SIMDType>::Int hasha13=SIMD<_SIMDType>::_and(hash, Constant::numi_13);

        //if h < 8 then x, else y
        typename SIMD<_SIMDType>::Mask l8=SIMD<_SIMDType>::lessThan(hasha13, Constant::numi_8);
        typename SIMD<_SIMDType>::Float u=SIMD<_SIMDType>::blend(y, x, l8);

        //if h < 4 then y else if h is 12 or 14 then x else z
        typename SIMD<_SIMDType>::Mask l4=SIMD<_SIMDType>::lessThan(hasha13, Constant::numi_2);
        typename SIMD<_SIMDType>::Mask h12o14=SIMD<_SIMDType>::equal(Constant::numi_12, hasha13);
        typename SIMD<_SIMDType>::Float v=SIMD<_SIMDType>::blend(SIMD<_SIMDType>::blend(z, x, h12o14), y, l4);

        //if h1 then -u else u
        //if h2 then -v else v
        typename SIMD<_SIMDType>::Float h1=SIMD<_SIMDType>::cast(SIMD<_SIMDType>::shiftL(hash, 31));
        typename SIMD<_SIMDType>::Float h2=SIMD<_SIMDType>::cast(SIMD<_SIMDType>::shiftL(SIMD<_SIMDType>::_and(hash, Constant::numi_2), 30));
        //then add them
        return SIMD<_SIMDType>::add(SIMD<_SIMDType>::_xor(u, h1), SIMD<_SIMDType>::_xor(v, h2));
    }
};

template<SIMDType _SIMDType, NoiseType _NoiseType>
struct Single
{
    static typename SIMD<_SIMDType>::Float VECTORCALL _(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z)
    {
        assert(false);
        return SIMD<_SIMDType>::zeroFloat();
    }
};

template<SIMDType _SIMDType>
struct Single<_SIMDType, NoiseType::WhiteNoise>
{
    static typename SIMD<_SIMDType>::Float VECTORCALL _(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        return ValCoord<_SIMDType>(seed,
            SIMD<_SIMDType>::mul(SIMD<_SIMDType>::_xor(SIMD<_SIMDType>::cast(x), SIMD<_SIMDType>::shiftR(SIMD<_SIMDType>::cast(x), 16)), Constant::numi_xPrime),
            SIMD<_SIMDType>::mul(SIMD<_SIMDType>::_xor(SIMD<_SIMDType>::cast(y), SIMD<_SIMDType>::shiftR(SIMD<_SIMDType>::cast(y), 16)), Constant::numi_yPrime),
            SIMD<_SIMDType>::mul(SIMD<_SIMDType>::_xor(SIMD<_SIMDType>::cast(z), SIMD<_SIMDType>::shiftR(SIMD<_SIMDType>::cast(z), 16)), Constant::numi_zPrime));
    }
};

template<SIMDType _SIMDType>
struct Single<_SIMDType, NoiseType::Value>
{
    static typename SIMD<_SIMDType>::Float VECTORCALL _(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        typename SIMD<_SIMDType>::Float xs=SIMD<_SIMDType>::floor(x);
        typename SIMD<_SIMDType>::Float ys=SIMD<_SIMDType>::floor(y);
        typename SIMD<_SIMDType>::Float zs=SIMD<_SIMDType>::floor(z);

        typename SIMD<_SIMDType>::Int x0=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(xs), Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y0=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(ys), Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z0=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(zs), Constant::numi_zPrime);
        typename SIMD<_SIMDType>::Int x1=SIMD<_SIMDType>::add(x0, Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y1=SIMD<_SIMDType>::add(y0, Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z1=SIMD<_SIMDType>::add(z0, Constant::numi_zPrime);

        xs=InterpQuintic<_SIMDType>(SIMD<_SIMDType>::sub(x, xs));
        ys=InterpQuintic<_SIMDType>(SIMD<_SIMDType>::sub(y, ys));
        zs=InterpQuintic<_SIMDType>(SIMD<_SIMDType>::sub(z, zs));

        return Lerp<_SIMDType>(
            Lerp<_SIMDType>(
                Lerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y0, z0), ValCoord<_SIMDType>(seed, x1, y0, z0), xs),
                Lerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y1, z0), ValCoord<_SIMDType>(seed, x1, y1, z0), xs), ys),
            Lerp<_SIMDType>(
                Lerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y0, z1), ValCoord<_SIMDType>(seed, x1, y0, z1), xs),
                Lerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y1, z1), ValCoord<_SIMDType>(seed, x1, y1, z1), xs), ys), zs);
    }
};

template<SIMDType _SIMDType>
struct Single<_SIMDType, NoiseType::Perlin>
{
    static typename SIMD<_SIMDType>::Float VECTORCALL _(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        typename SIMD<_SIMDType>::Float xs=SIMD<_SIMDType>::floor(x);
        typename SIMD<_SIMDType>::Float ys=SIMD<_SIMDType>::floor(y);
        typename SIMD<_SIMDType>::Float zs=SIMD<_SIMDType>::floor(z);

        typename SIMD<_SIMDType>::Int x0=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(xs), Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y0=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(ys), Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z0=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(zs), Constant::numi_zPrime);
        typename SIMD<_SIMDType>::Int x1=SIMD<_SIMDType>::add(x0, Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y1=SIMD<_SIMDType>::add(y0, Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z1=SIMD<_SIMDType>::add(z0, Constant::numi_zPrime);

        typename SIMD<_SIMDType>::Float xf0=xs=SIMD<_SIMDType>::sub(x, xs);
        typename SIMD<_SIMDType>::Float yf0=ys=SIMD<_SIMDType>::sub(y, ys);
        typename SIMD<_SIMDType>::Float zf0=zs=SIMD<_SIMDType>::sub(z, zs);
        typename SIMD<_SIMDType>::Float xf1=SIMD<_SIMDType>::sub(xf0, Constant::numf_1);
        typename SIMD<_SIMDType>::Float yf1=SIMD<_SIMDType>::sub(yf0, Constant::numf_1);
        typename SIMD<_SIMDType>::Float zf1=SIMD<_SIMDType>::sub(zf0, Constant::numf_1);

        xs=InterpQuintic<_SIMDType>(xs);
        ys=InterpQuintic<_SIMDType>(ys);
        zs=InterpQuintic<_SIMDType>(zs);

        return Lerp<_SIMDType>(
            Lerp<_SIMDType>(
                Lerp<_SIMDType>(GradCoord<_SIMDType>::_(seed, x0, y0, z0, xf0, yf0, zf0), GradCoord<_SIMDType>::_(seed, x1, y0, z0, xf1, yf0, zf0), xs),
                Lerp<_SIMDType>(GradCoord<_SIMDType>::_(seed, x0, y1, z0, xf0, yf1, zf0), GradCoord<_SIMDType>::_(seed, x1, y1, z0, xf1, yf1, zf0), xs), ys),
            Lerp<_SIMDType>(
                Lerp<_SIMDType>(GradCoord<_SIMDType>::_(seed, x0, y0, z1, xf0, yf0, zf1), GradCoord<_SIMDType>::_(seed, x1, y0, z1, xf1, yf0, zf1), xs),
                Lerp<_SIMDType>(GradCoord<_SIMDType>::_(seed, x0, y1, z1, xf0, yf1, zf1), GradCoord<_SIMDType>::_(seed, x1, y1, z1, xf1, yf1, zf1), xs), ys), zs);
    }
};

template<SIMDType _SIMDType>
struct Single<_SIMDType, NoiseType::Simplex>
{
    static typename SIMD<_SIMDType>::Float VECTORCALL _(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        typename SIMD<_SIMDType>::Float f=SIMD<_SIMDType>::mulf(Constant::numf_F3, SIMD<_SIMDType>::add(SIMD<_SIMDType>::add(x, y), z));
        typename SIMD<_SIMDType>::Float x0=SIMD<_SIMDType>::floor(SIMD<_SIMDType>::add(x, f));
        typename SIMD<_SIMDType>::Float y0=SIMD<_SIMDType>::floor(SIMD<_SIMDType>::add(y, f));
        typename SIMD<_SIMDType>::Float z0=SIMD<_SIMDType>::floor(SIMD<_SIMDType>::add(z, f));

        typename SIMD<_SIMDType>::Int i=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(x0), Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int j=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(y0), Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int k=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(z0), Constant::numi_zPrime);

        typename SIMD<_SIMDType>::Float g=SIMD<_SIMDType>::mulf(Constant::numf_G3, SIMD<_SIMDType>::add(SIMD<_SIMDType>::add(x0, y0), z0));
        x0=SIMD<_SIMDType>::sub(x, SIMD<_SIMDType>::sub(x0, g));
        y0=SIMD<_SIMDType>::sub(y, SIMD<_SIMDType>::sub(y0, g));
        z0=SIMD<_SIMDType>::sub(z, SIMD<_SIMDType>::sub(z0, g));

        typename SIMD<_SIMDType>::Mask x0_ge_y0=SIMD<_SIMDType>::greaterEqual(x0, y0);
        typename SIMD<_SIMDType>::Mask y0_ge_z0=SIMD<_SIMDType>::greaterEqual(y0, z0);
        typename SIMD<_SIMDType>::Mask x0_ge_z0=SIMD<_SIMDType>::greaterEqual(x0, z0);

        typename SIMD<_SIMDType>::Mask i1=SIMD<_SIMDType>::maskAnd(x0_ge_y0, x0_ge_z0);
        typename SIMD<_SIMDType>::Mask j1=SIMD<_SIMDType>::maskAndNot(x0_ge_y0, y0_ge_z0);
        typename SIMD<_SIMDType>::Mask k1=SIMD<_SIMDType>::maskAndNot(x0_ge_z0, SIMD<_SIMDType>::maskNot(y0_ge_z0));

        typename SIMD<_SIMDType>::Mask i2=SIMD<_SIMDType>::maskOr(x0_ge_y0, x0_ge_z0);
        typename SIMD<_SIMDType>::Mask j2=SIMD<_SIMDType>::maskOr(SIMD<_SIMDType>::maskNot(x0_ge_y0), y0_ge_z0);
        typename SIMD<_SIMDType>::Mask k2=SIMD<_SIMDType>::maskNot(SIMD<_SIMDType>::maskAnd(x0_ge_z0, y0_ge_z0));

        typename SIMD<_SIMDType>::Float x1=SIMD<_SIMDType>::add(SIMD<_SIMDType>::maskSub(i1, x0, Constant::numf_1), Constant::numf_G3);
        typename SIMD<_SIMDType>::Float y1=SIMD<_SIMDType>::add(SIMD<_SIMDType>::maskSub(j1, y0, Constant::numf_1), Constant::numf_G3);
        typename SIMD<_SIMDType>::Float z1=SIMD<_SIMDType>::add(SIMD<_SIMDType>::maskSub(k1, z0, Constant::numf_1), Constant::numf_G3);
        typename SIMD<_SIMDType>::Float x2=SIMD<_SIMDType>::add(SIMD<_SIMDType>::maskSub(i2, x0, Constant::numf_1), Constant::numf_F3);
        typename SIMD<_SIMDType>::Float y2=SIMD<_SIMDType>::add(SIMD<_SIMDType>::maskSub(j2, y0, Constant::numf_1), Constant::numf_F3);
        typename SIMD<_SIMDType>::Float z2=SIMD<_SIMDType>::add(SIMD<_SIMDType>::maskSub(k2, z0, Constant::numf_1), Constant::numf_F3);
        typename SIMD<_SIMDType>::Float x3=SIMD<_SIMDType>::add(x0, Constant::numf_G33);
        typename SIMD<_SIMDType>::Float y3=SIMD<_SIMDType>::add(y0, Constant::numf_G33);
        typename SIMD<_SIMDType>::Float z3=SIMD<_SIMDType>::add(z0, Constant::numf_G33);

        typename SIMD<_SIMDType>::Float t0=SIMD<_SIMDType>::nmulAdd(z0, z0, SIMD<_SIMDType>::nmulAdd(y0, y0, SIMD<_SIMDType>::nmulAdd(x0, x0, Constant::numf_0_6)));
        typename SIMD<_SIMDType>::Float t1=SIMD<_SIMDType>::nmulAdd(z1, z1, SIMD<_SIMDType>::nmulAdd(y1, y1, SIMD<_SIMDType>::nmulAdd(x1, x1, Constant::numf_0_6)));
        typename SIMD<_SIMDType>::Float t2=SIMD<_SIMDType>::nmulAdd(z2, z2, SIMD<_SIMDType>::nmulAdd(y2, y2, SIMD<_SIMDType>::nmulAdd(x2, x2, Constant::numf_0_6)));
        typename SIMD<_SIMDType>::Float t3=SIMD<_SIMDType>::nmulAdd(z3, z3, SIMD<_SIMDType>::nmulAdd(y3, y3, SIMD<_SIMDType>::nmulAdd(x3, x3, Constant::numf_0_6)));

        typename SIMD<_SIMDType>::Mask n0=SIMD<_SIMDType>::greaterEqual(t0, Constant::numf_0);
        typename SIMD<_SIMDType>::Mask n1=SIMD<_SIMDType>::greaterEqual(t1, Constant::numf_0);
        typename SIMD<_SIMDType>::Mask n2=SIMD<_SIMDType>::greaterEqual(t2, Constant::numf_0);
        typename SIMD<_SIMDType>::Mask n3=SIMD<_SIMDType>::greaterEqual(t3, Constant::numf_0);

        t0=SIMD<_SIMDType>::mulf(t0, t0);
        t1=SIMD<_SIMDType>::mulf(t1, t1);
        t2=SIMD<_SIMDType>::mulf(t2, t2);
        t3=SIMD<_SIMDType>::mulf(t3, t3);

        typename SIMD<_SIMDType>::Float v0=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::mulf(t0, t0), GradCoord<_SIMDType>::_(seed, i, j, k, x0, y0, z0));
        typename SIMD<_SIMDType>::Float v1=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::mulf(t1, t1), GradCoord<_SIMDType>::_(seed, SIMD<_SIMDType>::maskAdd(i1, i, Constant::numi_xPrime), SIMD<_SIMDType>::maskAdd(j1, j, Constant::numi_yPrime), SIMD<_SIMDType>::maskAdd(k1, k, Constant::numi_zPrime), x1, y1, z1));
        typename SIMD<_SIMDType>::Float v2=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::mulf(t2, t2), GradCoord<_SIMDType>::_(seed, SIMD<_SIMDType>::maskAdd(i2, i, Constant::numi_xPrime), SIMD<_SIMDType>::maskAdd(j2, j, Constant::numi_yPrime), SIMD<_SIMDType>::maskAdd(k2, k, Constant::numi_zPrime), x2, y2, z2));
        typename SIMD<_SIMDType>::Float v3=SIMD<_SIMDType>::mask(n3, SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::mulf(t3, t3), GradCoord<_SIMDType>::_(seed, SIMD<_SIMDType>::add(i, Constant::numi_xPrime), SIMD<_SIMDType>::add(j, Constant::numi_yPrime), SIMD<_SIMDType>::add(k, Constant::numi_zPrime), x3, y3, z3)));

        return SIMD<_SIMDType>::mulf(Constant::numf_32, SIMD<_SIMDType>::maskAdd(n0, SIMD<_SIMDType>::maskAdd(n1, SIMD<_SIMDType>::maskAdd(n2, v3, v2), v1), v0));
    }
};

template<SIMDType _SIMDType>
struct Single<_SIMDType, NoiseType::Cubic>
{
    static typename SIMD<_SIMDType>::Float VECTORCALL _(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
        typename SIMD<_SIMDType>::Float xf1=SIMD<_SIMDType>::floor(x);
        typename SIMD<_SIMDType>::Float yf1=SIMD<_SIMDType>::floor(y);
        typename SIMD<_SIMDType>::Float zf1=SIMD<_SIMDType>::floor(z);

        typename SIMD<_SIMDType>::Int x1=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(xf1), Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y1=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(yf1), Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z1=SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(zf1), Constant::numi_zPrime);

        typename SIMD<_SIMDType>::Int x0=SIMD<_SIMDType>::sub(x1, Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y0=SIMD<_SIMDType>::sub(y1, Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z0=SIMD<_SIMDType>::sub(z1, Constant::numi_zPrime);
        typename SIMD<_SIMDType>::Int x2=SIMD<_SIMDType>::add(x1, Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y2=SIMD<_SIMDType>::add(y1, Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z2=SIMD<_SIMDType>::add(z1, Constant::numi_zPrime);
        typename SIMD<_SIMDType>::Int x3=SIMD<_SIMDType>::add(x2, Constant::numi_xPrime);
        typename SIMD<_SIMDType>::Int y3=SIMD<_SIMDType>::add(y2, Constant::numi_yPrime);
        typename SIMD<_SIMDType>::Int z3=SIMD<_SIMDType>::add(z2, Constant::numi_zPrime);

        typename SIMD<_SIMDType>::Float xs=SIMD<_SIMDType>::sub(x, xf1);
        typename SIMD<_SIMDType>::Float ys=SIMD<_SIMDType>::sub(y, yf1);
        typename SIMD<_SIMDType>::Float zs=SIMD<_SIMDType>::sub(z, zf1);

        return SIMD<_SIMDType>::mulf(CubicLerp<_SIMDType>(
            CubicLerp<_SIMDType>(
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y0, z0), ValCoord<_SIMDType>(seed, x1, y0, z0), ValCoord<_SIMDType>(seed, x2, y0, z0), ValCoord<_SIMDType>(seed, x3, y0, z0), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y1, z0), ValCoord<_SIMDType>(seed, x1, y1, z0), ValCoord<_SIMDType>(seed, x2, y1, z0), ValCoord<_SIMDType>(seed, x3, y1, z0), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y2, z0), ValCoord<_SIMDType>(seed, x1, y2, z0), ValCoord<_SIMDType>(seed, x2, y2, z0), ValCoord<_SIMDType>(seed, x3, y2, z0), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y3, z0), ValCoord<_SIMDType>(seed, x1, y3, z0), ValCoord<_SIMDType>(seed, x2, y3, z0), ValCoord<_SIMDType>(seed, x3, y3, z0), xs),
                ys),
            CubicLerp<_SIMDType>(
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y0, z1), ValCoord<_SIMDType>(seed, x1, y0, z1), ValCoord<_SIMDType>(seed, x2, y0, z1), ValCoord<_SIMDType>(seed, x3, y0, z1), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y1, z1), ValCoord<_SIMDType>(seed, x1, y1, z1), ValCoord<_SIMDType>(seed, x2, y1, z1), ValCoord<_SIMDType>(seed, x3, y1, z1), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y2, z1), ValCoord<_SIMDType>(seed, x1, y2, z1), ValCoord<_SIMDType>(seed, x2, y2, z1), ValCoord<_SIMDType>(seed, x3, y2, z1), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y3, z1), ValCoord<_SIMDType>(seed, x1, y3, z1), ValCoord<_SIMDType>(seed, x2, y3, z1), ValCoord<_SIMDType>(seed, x3, y3, z1), xs),
                ys),
            CubicLerp<_SIMDType>(
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y0, z2), ValCoord<_SIMDType>(seed, x1, y0, z2), ValCoord<_SIMDType>(seed, x2, y0, z2), ValCoord<_SIMDType>(seed, x3, y0, z2), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y1, z2), ValCoord<_SIMDType>(seed, x1, y1, z2), ValCoord<_SIMDType>(seed, x2, y1, z2), ValCoord<_SIMDType>(seed, x3, y1, z2), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y2, z2), ValCoord<_SIMDType>(seed, x1, y2, z2), ValCoord<_SIMDType>(seed, x2, y2, z2), ValCoord<_SIMDType>(seed, x3, y2, z2), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y3, z2), ValCoord<_SIMDType>(seed, x1, y3, z2), ValCoord<_SIMDType>(seed, x2, y3, z2), ValCoord<_SIMDType>(seed, x3, y3, z2), xs),
                ys),
            CubicLerp<_SIMDType>(
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y0, z3), ValCoord<_SIMDType>(seed, x1, y0, z3), ValCoord<_SIMDType>(seed, x2, y0, z3), ValCoord<_SIMDType>(seed, x3, y0, z3), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y1, z3), ValCoord<_SIMDType>(seed, x1, y1, z3), ValCoord<_SIMDType>(seed, x2, y1, z3), ValCoord<_SIMDType>(seed, x3, y1, z3), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y2, z3), ValCoord<_SIMDType>(seed, x1, y2, z3), ValCoord<_SIMDType>(seed, x2, y2, z3), ValCoord<_SIMDType>(seed, x3, y2, z3), xs),
                CubicLerp<_SIMDType>(ValCoord<_SIMDType>(seed, x0, y3, z3), ValCoord<_SIMDType>(seed, x1, y3, z3), ValCoord<_SIMDType>(seed, x2, y3, z3), ValCoord<_SIMDType>(seed, x3, y3, z3), xs),
                ys),
            zs), Constant::numf_cubicBounding);
    }
};

//#define GRADIENT_COORD(_x,_y,_z)
template<SIMDType _SIMDType>
void gradientCoord(typename SIMD<_SIMDType>::Int seed, const typename SIMD<_SIMDType>::Int &x, const typename SIMD<_SIMDType>::Int &y, const typename SIMD<_SIMDType>::Int &z,
    typename SIMD<_SIMDType>::Float &xGrad, typename SIMD<_SIMDType>::Float &yGrad, typename SIMD<_SIMDType>::Float &zGrad)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
    typename SIMD<_SIMDType>::Int hash=HashHB<_SIMDType>(seed, x, y, z);

    xGrad = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(hash, Constant::numi_bit10Mask)), Constant::numf_511_5);
    yGrad = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash, 10), Constant::numi_bit10Mask)), Constant::numf_511_5);
    zGrad = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash, 20), Constant::numi_bit10Mask)), Constant::numf_511_5); 
}
//typename SIMD<_SIMDType>::Float invMag##_x##_y##_z = SIMD<_SIMDType>::mul(Constant::numf_cellJitter, SIMD<_SIMDType>::invSqrt(SIMD<_SIMDType>::mulAdd(x##_x##_y##_z, x##_x##_y##_z, SIMD<_SIMDType>::mulAdd(y##_x##_y##_z, y##_x##_y##_z, SIMD<_SIMDType>::mul(z##_x##_y##_z, z##_x##_y##_z)))));
//x##_x##_y##_z = SIMD<_SIMDType>::mul(x##_x##_y##_z, invMag##_x##_y##_z);
//y##_x##_y##_z = SIMD<_SIMDType>::mul(y##_x##_y##_z, invMag##_x##_y##_z); 
//z##_x##_y##_z = SIMD<_SIMDType>::mul(z##_x##_y##_z, invMag##_x##_y##_z);

template<SIMDType _SIMDType>
static void VECTORCALL GradientPerturbSingle(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float perturbAmp, typename SIMD<_SIMDType>::Float perturbFrequency, typename SIMD<_SIMDType>::Float& x, typename SIMD<_SIMDType>::Float& y, typename SIMD<_SIMDType>::Float& z)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
	typename SIMD<_SIMDType>::Float xf = SIMD<_SIMDType>::mulf(x, perturbFrequency);
	typename SIMD<_SIMDType>::Float yf = SIMD<_SIMDType>::mulf(y, perturbFrequency);
	typename SIMD<_SIMDType>::Float zf = SIMD<_SIMDType>::mulf(z, perturbFrequency);

	typename SIMD<_SIMDType>::Float xs = SIMD<_SIMDType>::floor(xf);
	typename SIMD<_SIMDType>::Float ys = SIMD<_SIMDType>::floor(yf);
	typename SIMD<_SIMDType>::Float zs = SIMD<_SIMDType>::floor(zf);

	typename SIMD<_SIMDType>::Int x0 = SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(xs), Constant::numi_xPrime);
	typename SIMD<_SIMDType>::Int y0 = SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(ys), Constant::numi_yPrime);
	typename SIMD<_SIMDType>::Int z0 = SIMD<_SIMDType>::mul(SIMD<_SIMDType>::convert(zs), Constant::numi_zPrime);
	typename SIMD<_SIMDType>::Int x1 = SIMD<_SIMDType>::add(x0, Constant::numi_xPrime);
	typename SIMD<_SIMDType>::Int y1 = SIMD<_SIMDType>::add(y0, Constant::numi_yPrime);
	typename SIMD<_SIMDType>::Int z1 = SIMD<_SIMDType>::add(z0, Constant::numi_zPrime);

	xs = InterpQuintic<_SIMDType>(SIMD<_SIMDType>::sub(xf, xs));
	ys = InterpQuintic<_SIMDType>(SIMD<_SIMDType>::sub(yf, ys));
	zs = InterpQuintic<_SIMDType>(SIMD<_SIMDType>::sub(zf, zs));

    typename SIMD<_SIMDType>::Float x000, y000, z000;
    typename SIMD<_SIMDType>::Float x001, y001, z001;
    typename SIMD<_SIMDType>::Float x010, y010, z010;
    typename SIMD<_SIMDType>::Float x011, y011, z011;
    typename SIMD<_SIMDType>::Float x100, y100, z100;
    typename SIMD<_SIMDType>::Float x101, y101, z101;
    typename SIMD<_SIMDType>::Float x110, y110, z110;
    typename SIMD<_SIMDType>::Float x111, y111, z111;

	gradientCoord<_SIMDType>(seed, x0, y0, z0, x000, y000, z000);
    gradientCoord<_SIMDType>(seed, x0, y0, z1, x001, y001, z001);
    gradientCoord<_SIMDType>(seed, x0, y1, z0, x010, y010, z010);
    gradientCoord<_SIMDType>(seed, x0, y1, z1, x011, y011, z011);
    gradientCoord<_SIMDType>(seed, x1, y0, z0, x100, y100, z100);
    gradientCoord<_SIMDType>(seed, x1, y0, z1, x101, y101, z101);
    gradientCoord<_SIMDType>(seed, x1, y1, z0, x110, y110, z110);
    gradientCoord<_SIMDType>(seed, x1, y1, z1, x111, y111, z111);

	typename SIMD<_SIMDType>::Float x0y = Lerp<_SIMDType>(Lerp<_SIMDType>(x000, x100, xs), Lerp<_SIMDType>(x010, x110, xs), ys);
	typename SIMD<_SIMDType>::Float y0y = Lerp<_SIMDType>(Lerp<_SIMDType>(y000, y100, xs), Lerp<_SIMDType>(y010, y110, xs), ys);
	typename SIMD<_SIMDType>::Float z0y = Lerp<_SIMDType>(Lerp<_SIMDType>(z000, z100, xs), Lerp<_SIMDType>(z010, z110, xs), ys);

	typename SIMD<_SIMDType>::Float x1y = Lerp<_SIMDType>(Lerp<_SIMDType>(x001, x101, xs), Lerp<_SIMDType>(x011, x111, xs), ys);
	typename SIMD<_SIMDType>::Float y1y = Lerp<_SIMDType>(Lerp<_SIMDType>(y001, y101, xs), Lerp<_SIMDType>(y011, y111, xs), ys);
	typename SIMD<_SIMDType>::Float z1y = Lerp<_SIMDType>(Lerp<_SIMDType>(z001, z101, xs), Lerp<_SIMDType>(z011, z111, xs), ys);

	x = SIMD<_SIMDType>::mulAdd(Lerp<_SIMDType>(x0y, x1y, zs), perturbAmp, x);
	y = SIMD<_SIMDType>::mulAdd(Lerp<_SIMDType>(y0y, y1y, zs), perturbAmp, y);
	z = SIMD<_SIMDType>::mulAdd(Lerp<_SIMDType>(z0y, z1y, zs), perturbAmp, z);
}

template<SIMDType _SIMDType>
NoiseSIMD<_SIMDType>::NoiseSIMD(int seed)
{
    InitSIMDValues<_SIMDType>::_();

    m_noiseDetails.seed = seed;
    m_noiseDetails.fractalBounding = CalculateFractalBounding(m_noiseDetails.octaves, m_noiseDetails.gain);
	m_perturbDetails.FractalBounding = CalculateFractalBounding(m_perturbDetails.Octaves, m_perturbDetails.Gain);

	s_currentSIMDLevel = SIMD<_SIMDType>::level();
}

template<SIMDType _SIMDType>
FastNoise::NoiseSIMD *NoiseSIMD<_SIMDType>::create(int seed)
{
    return new NoiseSIMD<_SIMDType>(seed);
}

template<SIMDType _SIMDType>
size_t NoiseSIMD<_SIMDType>::AlignedSize(size_t size)
{
#ifdef FN_ALIGNED_SETS
	// size must be a multiple of SIMD<_SIMDType>::vectorSize() (8)
	if ((size & (SIMD<_SIMDType>::vectorSize() - 1)) != 0)
	{
		size &= ~(SIMD<_SIMDType>::vectorSize() - 1);
		size += SIMD<_SIMDType>::vectorSize();
	}
#endif
	return size;
}

template<SIMDType _SIMDType>
float* NoiseSIMD<_SIMDType>::GetEmptySet(size_t size)
{
	size = AlignedSize(size);

	float* noiseSet;
//	SIMD_ALLOCATE_SET(noiseSet, size);
    noiseSet=simdAlloc<_SIMDType>::_(size);

	return noiseSet;
}

//#define AXIS_RESET(_zSize, _start)
template<SIMDType _SIMDType>
void axisReset(typename SIMD<_SIMDType>::Int &x, typename SIMD<_SIMDType>::Int &y, typename SIMD<_SIMDType>::Int &z, 
    const typename SIMD<_SIMDType>::Int &ySizeV, const typename SIMD<_SIMDType>::Int &yEndV, const typename SIMD<_SIMDType>::Int &zSizeV, const typename SIMD<_SIMDType>::Int &zEndV, int &_zSize, int _start)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

    for(int _i=(_zSize) * (_start); _i<SIMD<_SIMDType>::vectorSize(); _i+=(_zSize))
    {
        typename SIMD<_SIMDType>::Mask _zReset=SIMD<_SIMDType>::greaterThan(z, zEndV);

        y=SIMD<_SIMDType>::maskAdd(_zReset, y, Constant::numi_1);
        z=SIMD<_SIMDType>::maskSub(_zReset, z, zSizeV);
            
        typename SIMD<_SIMDType>::Mask _yReset=SIMD<_SIMDType>::greaterThan(y, yEndV);

        x=SIMD<_SIMDType>::maskAdd(_yReset, x, Constant::numi_1);
        y=SIMD<_SIMDType>::maskSub(_yReset, y, ySizeV);
    }
}

#ifdef FN_ALIGNED_SETS
#define STORE_LAST_RESULT(_dest, _source) SIMD<_SIMDType>::store(_dest, _source)
#else
#include <cstring>
#define STORE_LAST_RESULT(_dest, _source) std::memcpy(_dest, &_source, (maxIndex - index) * 4)
#endif

template<SIMDType _SIMDType>
struct NoiseValues
{
    typename SIMD<_SIMDType>::Int seedV;
    typename SIMD<_SIMDType>::Float lacunarityV;
    typename SIMD<_SIMDType>::Float gainV;
    typename SIMD<_SIMDType>::Float fractalBoundingV;

    typename SIMD<_SIMDType>::Float xFreqV;
    typename SIMD<_SIMDType>::Float yFreqV;
    typename SIMD<_SIMDType>::Float zFreqV;

    typename SIMD<_SIMDType>::Float cellJitterV;

    size_t octaves;
};

template<SIMDType _SIMDType>
NoiseValues<_SIMDType> initNoise(const NoiseDetails &noiseDetails, float scaleModifier=1.0f)
{
    NoiseValues<_SIMDType> noise;

    noise.seedV=SIMD<_SIMDType>::set(noiseDetails.seed);
    noise.lacunarityV=SIMD<_SIMDType>::set(noiseDetails.lacunarity);
    noise.gainV=SIMD<_SIMDType>::set(noiseDetails.gain);
    noise.fractalBoundingV=SIMD<_SIMDType>::set(noiseDetails.fractalBounding);

    scaleModifier*=noiseDetails.frequency;
    noise.xFreqV=SIMD<_SIMDType>::set(scaleModifier * noiseDetails.xScale);
    noise.yFreqV=SIMD<_SIMDType>::set(scaleModifier * noiseDetails.yScale);
    noise.zFreqV=SIMD<_SIMDType>::set(scaleModifier * noiseDetails.zScale);

    noise.cellJitterV=SIMD<_SIMDType>::set(noiseDetails.cellularJitter);

    noise.octaves=noiseDetails.octaves;
    return noise;
};

template<SIMDType _SIMDType, PerturbType _PerturbType>
struct init_perturb
{
    static PerturbValues<_SIMDType> _(const NoiseDetails &noiseDetails, const PerturbDetails &perturbDetails)
    {
        PerturbValues<_SIMDType> perturb;

        perturb.Octaves=perturbDetails.Octaves;
        return perturb;
    }
};

template<SIMDType _SIMDType>
struct init_perturb<_SIMDType, PerturbType::Gradient_Normalise>
{
    static PerturbValues<_SIMDType> _(const NoiseDetails &noiseDetails, const PerturbDetails &perturbDetails)
    {
        PerturbValues<_SIMDType> perturb;

        perturb.Octaves=perturbDetails.Octaves;
        perturb.NormaliseLengthV=SIMD<_SIMDType>::set(perturbDetails.NormaliseLength*noiseDetails.frequency);
        return perturb;
    }
};

template<SIMDType _SIMDType>
struct init_perturb<_SIMDType, PerturbType::Gradient>
{
    static PerturbValues<_SIMDType> _(const NoiseDetails &noiseDetails, const PerturbDetails &perturbDetails)
    {
        PerturbValues<_SIMDType> perturb;

        perturb.Octaves=perturbDetails.Octaves;
        perturb.AmpV=SIMD<_SIMDType>::set(perturbDetails.Amp);
        perturb.FreqV=SIMD<_SIMDType>::set(perturbDetails.Frequency);
        return perturb;
    }
};

template<SIMDType _SIMDType>
struct init_perturb<_SIMDType, PerturbType::GradientFractal_Normalise>
{
    static PerturbValues<_SIMDType> _(const NoiseDetails &noiseDetails, const PerturbDetails &perturbDetails)
    {
        PerturbValues<_SIMDType> perturb;

        perturb.Octaves=perturbDetails.Octaves;
        perturb.NormaliseLengthV=SIMD<_SIMDType>::set(perturbDetails.NormaliseLength*noiseDetails.frequency);
        return perturb;
    }
};

template<SIMDType _SIMDType>
struct init_perturb<_SIMDType, PerturbType::GradientFractal>
{
    static PerturbValues<_SIMDType> _(const NoiseDetails &noiseDetails, const PerturbDetails &perturbDetails)
    {
        PerturbValues<_SIMDType> perturb;

        perturb.Octaves=perturbDetails.Octaves;
        perturb.AmpV=SIMD<_SIMDType>::set(perturbDetails.Amp*noiseDetails.fractalBounding);
        perturb.FreqV=SIMD<_SIMDType>::set(perturbDetails.Frequency);
        perturb.LacunarityV=SIMD<_SIMDType>::set(perturbDetails.Lacunarity);
        perturb.GainV=SIMD<_SIMDType>::set(perturbDetails.Gain);
        return perturb;
    }
};

template<SIMDType _SIMDType>
struct init_perturb<_SIMDType, PerturbType::Normalise>
{
    static PerturbValues<_SIMDType> _(const NoiseDetails &noiseDetails, const PerturbDetails &perturbDetails)
    {
        PerturbValues<_SIMDType> perturb;

        perturb.Octaves=perturbDetails.Octaves;
        perturb.NormaliseLengthV=SIMD<_SIMDType>::set(perturbDetails.NormaliseLength*noiseDetails.frequency);
        return perturb;
    }
};

template<SIMDType _SIMDType>
PerturbValues<_SIMDType> initPerturb(PerturbType perturbType, const NoiseDetails &noiseDetails, const PerturbDetails &perturbDetails)
{
    switch(perturbType)
    {
    case PerturbType::None:
        return init_perturb<_SIMDType, PerturbType::None>::_(noiseDetails, perturbDetails);
        break;
    case PerturbType::Gradient:
        return init_perturb<_SIMDType, PerturbType::Gradient>::_(noiseDetails, perturbDetails);
        break;
    case PerturbType::GradientFractal:
        return init_perturb<_SIMDType, PerturbType::GradientFractal>::_(noiseDetails, perturbDetails);
        break;
    case PerturbType::Normalise:
        return init_perturb<_SIMDType, PerturbType::Normalise>::_(noiseDetails, perturbDetails);
        break;
    case PerturbType::Gradient_Normalise:
        return init_perturb<_SIMDType, PerturbType::Gradient_Normalise>::_(noiseDetails, perturbDetails);
        break;
    case PerturbType::GradientFractal_Normalise:
        return init_perturb<_SIMDType, PerturbType::GradientFractal_Normalise>::_(noiseDetails, perturbDetails);
        break;
    }

    return init_perturb<_SIMDType, PerturbType::None>::_(noiseDetails, perturbDetails);
};

template<SIMDType _SIMDType, PerturbType _PerturbType>
struct Perturb
{
    static void _(const typename SIMD<_SIMDType>::Int &seedV, const PerturbValues<_SIMDType> &perturb, const typename SIMD<_SIMDType>::Float &xF, const typename SIMD<_SIMDType>::Float &yF, const typename SIMD<_SIMDType>::Float &zF)
    {}
};

template<SIMDType _SIMDType>
struct Perturb<_SIMDType, PerturbType::Gradient>
{
    static void _(const typename SIMD<_SIMDType>::Int &seedV, const PerturbValues<_SIMDType> &perturb, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        GradientPerturbSingle<_SIMDType>(SIMD<_SIMDType>::sub(seedV, Constant::numi_1), perturb.AmpV, perturb.FreqV, xF, yF, zF);
    }
};

template<SIMDType _SIMDType>
struct Perturb<_SIMDType, PerturbType::GradientFractal>
{
    static void _(const typename SIMD<_SIMDType>::Int &seedV, const PerturbValues<_SIMDType> &perturb, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        typename SIMD<_SIMDType>::Int seedF=SIMD<_SIMDType>::sub(seedV, Constant::numi_1);
        typename SIMD<_SIMDType>::Float freqF=perturb.FreqV;
        typename SIMD<_SIMDType>::Float ampF=perturb.AmpV;
    
        GradientPerturbSingle<_SIMDType>(seedF, ampF, freqF, xF, yF, zF);
    
        int octaveIndex=0;
    
        while(++octaveIndex < perturb.Octaves)
        {
            freqF=SIMD<_SIMDType>::mulf(freqF, perturb.LacunarityV);
            seedF=SIMD<_SIMDType>::sub(seedF, Constant::numi_1);
            ampF=SIMD<_SIMDType>::mulf(ampF, perturb.GainV);
        
            GradientPerturbSingle<_SIMDType>(seedF, ampF, freqF, xF, yF, zF);
        }
    }
};

template<SIMDType _SIMDType>
struct Perturb<_SIMDType, PerturbType::Gradient_Normalise>
{
    static void _(const typename SIMD<_SIMDType>::Int &seedV, const PerturbValues<_SIMDType> &perturb, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        GradientPerturbSingle<_SIMDType>(SIMD<_SIMDType>::sub(seedV, Constant::numi_1), perturb.AmpV, perturb.FreqV, xF, yF, zF);
    }
};

template<SIMDType _SIMDType>
struct Perturb<_SIMDType, PerturbType::Normalise>
{
    static void _(const typename SIMD<_SIMDType>::Int &seedV, const PerturbValues<_SIMDType> &perturb, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {
        typename SIMD<_SIMDType>::Float invMag=SIMD<_SIMDType>::mulf(perturb.NormaliseLengthV, SIMD<_SIMDType>::invSqrt(SIMD<_SIMDType>::mulAdd(xF, xF, SIMD<_SIMDType>::mulAdd(yF, yF, SIMD<_SIMDType>::mulf(zF, zF)))));

        xF=SIMD<_SIMDType>::mulf(xF, invMag);
        yF=SIMD<_SIMDType>::mulf(yF, invMag);
        zF=SIMD<_SIMDType>::mulf(zF, invMag);
    }
};

template<SIMDType _SIMDType>
struct Perturb<_SIMDType, PerturbType::GradientFractal_Normalise>
{
    static void _(const typename SIMD<_SIMDType>::Int &seedV, const PerturbValues<_SIMDType> &perturb, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {
        typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

        typename SIMD<_SIMDType>::Int seedF=SIMD<_SIMDType>::sub(seedV, Constant::numi_1);
        typename SIMD<_SIMDType>::Float freqF=perturb.FreqV;
        typename SIMD<_SIMDType>::Float ampF=perturb.AmpV;
    
        GradientPerturbSingle<_SIMDType>(seedF, ampF, freqF, xF, yF, zF);
    
        int octaveIndex=0; 
    
        while(++octaveIndex < perturb.Octaves)
        {
            freqF=SIMD<_SIMDType>::mulf(freqF, perturb.LacunarityV);
            seedF=SIMD<_SIMDType>::sub(seedF, Constant::numi_1);
            ampF=SIMD<_SIMDType>::mulf(ampF, perturb.GainV);
        
            GradientPerturbSingle<_SIMDType>(seedF, ampF, freqF, xF, yF, zF);
        }
        typename SIMD<_SIMDType>::Float invMag=SIMD<_SIMDType>::mulf(perturb.NormaliseLengthV, SIMD<_SIMDType>::invSqrt(SIMD<_SIMDType>::mulAdd(xF, xF, SIMD<_SIMDType>::mulAdd(yF, yF, SIMD<_SIMDType>::mulf(zF, zF)))));
        xF=SIMD<_SIMDType>::mulf(xF, invMag);
        yF=SIMD<_SIMDType>::mulf(yF, invMag);
        zF=SIMD<_SIMDType>::mulf(zF, invMag);
    }
};

// FBM SINGLE
//#define FBM_SINGLE(f)
template<SIMDType _SIMDType, NoiseType _NoiseType>
typename SIMD<_SIMDType>::Float FBMSingle(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
    typename SIMD<_SIMDType>::Int seedF=noise.seedV;
    typename SIMD<_SIMDType>::Float result=Single<_SIMDType, _NoiseType>::_(seedF, xF, yF, zF);
    typename SIMD<_SIMDType>::Float ampF=Constant::numf_1;
    int octaveIndex=0;

    while(++octaveIndex<noise.octaves)
    {
        xF=SIMD<_SIMDType>::mulf(xF, noise.lacunarityV);
        yF=SIMD<_SIMDType>::mulf(yF, noise.lacunarityV);
        zF=SIMD<_SIMDType>::mulf(zF, noise.lacunarityV);
        seedF=SIMD<_SIMDType>::add(seedF, Constant::numi_1);

        ampF=SIMD<_SIMDType>::mulf(ampF, noise.gainV);
        result=SIMD<_SIMDType>::mulAdd(Single<_SIMDType, _NoiseType>::_(seedF, xF, yF, zF), ampF, result);
    }
    result=SIMD<_SIMDType>::mulf(result, noise.fractalBoundingV);

    return result;
}

// BILLOW SINGLE
//#define BILLOW_SINGLE(f)
template<SIMDType _SIMDType, NoiseType _NoiseType>
typename SIMD<_SIMDType>::Float BillowSingle(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

    typename SIMD<_SIMDType>::Int seedF=noise.seedV;
    typename SIMD<_SIMDType>::Float result=SIMD<_SIMDType>::mulSub(SIMD<_SIMDType>::abs(Single<_SIMDType, _NoiseType>::_(seedF, xF, yF, zF)), Constant::numf_2, Constant::numf_1);
    typename SIMD<_SIMDType>::Float ampF=Constant::numf_1;
    int octaveIndex=0;

    while(++octaveIndex<noise.octaves)
    {
        xF=SIMD<_SIMDType>::mulf(xF, noise.lacunarityV);
        yF=SIMD<_SIMDType>::mulf(yF, noise.lacunarityV);
        zF=SIMD<_SIMDType>::mulf(zF, noise.lacunarityV);
        seedF=SIMD<_SIMDType>::add(seedF, Constant::numi_1);

        ampF=SIMD<_SIMDType>::mulf(ampF, noise.gainV);
        result=SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::mulSub(SIMD<_SIMDType>::abs(Single<_SIMDType, _NoiseType>::_(seedF, xF, yF, zF)), Constant::numf_2, Constant::numf_1), ampF, result);
    }
    result=SIMD<_SIMDType>::mulf(result, noise.fractalBoundingV);

    return result;
}

// RIGIDMULTI SINGLE
//#define RIGIDMULTI_SINGLE(f)
template<SIMDType _SIMDType, NoiseType _NoiseType>
typename SIMD<_SIMDType>::Float RigidMultiSingle(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

    typename SIMD<_SIMDType>::Int seedF=noise.seedV;
    typename SIMD<_SIMDType>::Float result=SIMD<_SIMDType>::sub(Constant::numf_1, SIMD<_SIMDType>::abs(Single<_SIMDType, _NoiseType>::_(seedF, xF, yF, zF)));
    typename SIMD<_SIMDType>::Float ampF=Constant::numf_1;
    int octaveIndex=0;

    while(++octaveIndex < noise.octaves)
    {
        xF=SIMD<_SIMDType>::mulf(xF, noise.lacunarityV);
        yF=SIMD<_SIMDType>::mulf(yF, noise.lacunarityV);
        zF=SIMD<_SIMDType>::mulf(zF, noise.lacunarityV);
        seedF=SIMD<_SIMDType>::add(seedF, Constant::numi_1);

        ampF=SIMD<_SIMDType>::mulf(ampF, noise.gainV);
        result=SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::sub(Constant::numf_1, SIMD<_SIMDType>::abs(Single<_SIMDType, _NoiseType>::_(seedF, xF, yF, zF))), ampF, result);
    }
    return result;
}

template<SIMDType _SIMDType, NoiseType _NoiseType, FractalType _FractalType>
struct GetSingle
{
    template<typename ..._Types>
    static typename SIMD<_SIMDType>::Float _(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF, _Types ...args)
    { assert(false);  return SIMD<_SIMDType>::zeroFloat(); }
};

template<SIMDType _SIMDType, NoiseType _NoiseType>
struct GetSingle<_SIMDType, _NoiseType, FractalType::None>
{
    template<typename ..._Types>
    static typename SIMD<_SIMDType>::Float _(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF, _Types ...args)
    { return Single<_SIMDType, _NoiseType>::_(noise.seedV, xF, yF, zF); }
};

template<SIMDType _SIMDType, NoiseType _NoiseType>
struct GetSingle<_SIMDType, _NoiseType, FractalType::FBM>
{
    template<typename ..._Types>
    static typename SIMD<_SIMDType>::Float _(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {   return FBMSingle<_SIMDType, _NoiseType>(noise, xF, yF, zF); }
};

template<SIMDType _SIMDType, NoiseType _NoiseType>
struct GetSingle<_SIMDType, _NoiseType, FractalType::Billow>
{
    template<typename ..._Types>
    static typename SIMD<_SIMDType>::Float _(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {   return BillowSingle<_SIMDType, _NoiseType>(noise, xF, yF, zF); }
};

template<SIMDType _SIMDType, NoiseType _NoiseType>
struct GetSingle<_SIMDType, _NoiseType, FractalType::RigidMulti>
{
    template<typename ..._Types>
    static typename SIMD<_SIMDType>::Float _(const NoiseValues<_SIMDType> &noise, typename SIMD<_SIMDType>::Float &xF, typename SIMD<_SIMDType>::Float &yF, typename SIMD<_SIMDType>::Float &zF)
    {   return RigidMultiSingle<_SIMDType, _NoiseType>(noise, xF, yF, zF); }
};

//#define SET_BUILDER(f)
template<SIMDType _SIMDType, NoiseType _NoiseType, FractalType _FractalType, PerturbType _PerturbType, typename... _Types>
static void Build(const NoiseValues<_SIMDType> &noise, const PerturbValues<_SIMDType> &perturb,
    float *noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, _Types... args)
{
    typedef typename SIMD<_SIMDType>::Float Float;
    typedef typename SIMD<_SIMDType>::Int Int;
    typedef Constants<Float, Int, _SIMDType> Constant;

    if((zSize & (SIMD<_SIMDType>::vectorSize()-1))==0)
    {
        Int yBase=SIMD<_SIMDType>::set(yStart);
        Int zBase=SIMD<_SIMDType>::add(Constant::numi_incremental, SIMD<_SIMDType>::set(zStart));
        Int x=SIMD<_SIMDType>::set(xStart);

        int index=0;

        for(int ix=0; ix<xSize; ix++)
        {
            Float xf=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(x), noise.xFreqV);
            Int y=yBase;

            for(int iy=0; iy<ySize; iy++)
            {
                Float yf=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(y), noise.yFreqV);
                Int z=zBase;
                Float xF=xf;
                Float yF=yf;
                Float zF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(z), noise.zFreqV);

                //    			PERTURB_SWITCH()
                Perturb<_SIMDType, _PerturbType>::_(noise.seedV, perturb, xF, yF, zF);
                Float result=GetSingle<_SIMDType, _NoiseType, _FractalType>::_(noise, xF, yF, zF, args...);
                SIMD<_SIMDType>::store(&noiseSet[index], result);

                int iz=SIMD<_SIMDType>::vectorSize();
                while(iz<zSize)
                {
                    z=SIMD<_SIMDType>::add(z, Constant::numi_vectorSize);
                    index+=SIMD<_SIMDType>::vectorSize();
                    iz+=SIMD<_SIMDType>::vectorSize();
                    xF=xf;
                    yF=yf;
                    zF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(z), noise.zFreqV);

                    //    				PERTURB_SWITCH()
                    Perturb<_SIMDType, _PerturbType>::_(noise.seedV, perturb, xF, yF, zF);
                    Float result=GetSingle<_SIMDType, _NoiseType, _FractalType>::_(noise, xF, yF, zF, args...);
                    SIMD<_SIMDType>::store(&noiseSet[index], result);
                }
                index+=SIMD<_SIMDType>::vectorSize();
                y=SIMD<_SIMDType>::add(y, Constant::numi_1);
            }
            x=SIMD<_SIMDType>::add(x, Constant::numi_1);
        }
    }
    else
    {
        Int ySizeV=SIMD<_SIMDType>::set(ySize);
        Int zSizeV=SIMD<_SIMDType>::set(zSize);

        Int yEndV=SIMD<_SIMDType>::set(yStart+ySize-1);
        Int zEndV=SIMD<_SIMDType>::set(zStart+zSize-1);

        Int x=SIMD<_SIMDType>::set(xStart);
        Int y=SIMD<_SIMDType>::set(yStart);
        Int z=SIMD<_SIMDType>::add(SIMD<_SIMDType>::set(zStart), Constant::numi_incremental);
//        AXIS_RESET(zSize, 1)
        axisReset<_SIMDType>(x, y, z, ySizeV, yEndV, zSizeV, zEndV, zSize, 1);

            int index=0;
        int maxIndex=xSize * ySize * zSize;

        for(; index<maxIndex-SIMD<_SIMDType>::vectorSize(); index+=SIMD<_SIMDType>::vectorSize())
        {
            Float xF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(x), noise.xFreqV);
            Float yF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(y), noise.yFreqV);
            Float zF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(z), noise.zFreqV);

            //    		PERTURB_SWITCH()
            Perturb<_SIMDType, _PerturbType>::_(noise.seedV, perturb, xF, yF, zF);
            Float result=GetSingle<_SIMDType, _NoiseType, _FractalType>::_(noise, xF, yF, zF, args...);

            SIMD<_SIMDType>::store(&noiseSet[index], result);

            z=SIMD<_SIMDType>::add(z, Constant::numi_vectorSize);

//            AXIS_RESET(zSize, 0)
            axisReset<_SIMDType>(x, y, z, ySizeV, yEndV, zSizeV, zEndV, zSize, 0);
        }

        Float xF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(x), noise.xFreqV);
        Float yF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(y), noise.yFreqV);
        Float zF=SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::convert(z), noise.zFreqV);

        //    	PERTURB_SWITCH()
        Perturb<_SIMDType, _PerturbType>::_(noise.seedV, perturb, xF, yF, zF);
        Float result=GetSingle<_SIMDType, _NoiseType, _FractalType>::_(noise, xF, yF, zF, args...);
        STORE_LAST_RESULT(&noiseSet[index], result);
    }
}

template<SIMDType _SIMDType, FractalType _FractalType, PerturbType _PerturbType, typename... _Types>
static void CallBuild(NoiseType noiseType, _Types... args)
{
    switch(noiseType)
    {
    case NoiseType::Value:
    case NoiseType::ValueFractal:
        Build<_SIMDType, NoiseType::Value, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Perlin:
    case NoiseType::PerlinFractal:
        Build<_SIMDType, NoiseType::Perlin, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Simplex:
    case NoiseType::SimplexFractal:
        Build<_SIMDType, NoiseType::Simplex, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::WhiteNoise:
        Build<_SIMDType, NoiseType::WhiteNoise, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Cellular:
        Build<_SIMDType, NoiseType::Cellular, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Cubic:
    case NoiseType::CubicFractal:
        Build<_SIMDType, NoiseType::Cubic, _FractalType, _PerturbType>(args...);
        break;
    }
}

template<SIMDType _SIMDType, PerturbType _PerturbType, typename... _Types>
static void CallBuild(NoiseType noiseType, FractalType fractalType, _Types... args)
{
    FractalType type=FractalType::None;

    //only use fractal type if it is fractal noise
    if((noiseType==NoiseType::ValueFractal)||(noiseType==NoiseType::PerlinFractal)||(noiseType==NoiseType::SimplexFractal)||(noiseType==NoiseType::CubicFractal))
        type=fractalType;

    switch(type)
    {
    case FractalType::None:
        CallBuild<_SIMDType, FractalType::None, _PerturbType>(noiseType, args...);
        break;
    case FractalType::FBM:
        CallBuild<_SIMDType, FractalType::FBM, _PerturbType>(noiseType, args...);
        break;
    case FractalType::Billow:
        CallBuild<_SIMDType, FractalType::Billow, _PerturbType>(noiseType, args...);
        break;
    case FractalType::RigidMulti:
        CallBuild<_SIMDType, FractalType::RigidMulti, _PerturbType>(noiseType, args...);
        break;
    }
}

template<SIMDType _SIMDType, typename... _Types>
static void CallBuild(NoiseType noiseType, FractalType fractalType, PerturbType perturbType, _Types... args)
{
    switch(perturbType)
    {
    case PerturbType::None:
        CallBuild<_SIMDType, PerturbType::None>(noiseType, fractalType, args...);
        break;
    case PerturbType::Gradient:
        CallBuild<_SIMDType, PerturbType::Gradient>(noiseType, fractalType, args...);
        break;
    case PerturbType::GradientFractal:
        CallBuild<_SIMDType, PerturbType::GradientFractal>(noiseType, fractalType, args...);
        break;
    case PerturbType::Normalise:
        CallBuild<_SIMDType, PerturbType::Normalise>(noiseType, fractalType, args...);
        break;
    case PerturbType::Gradient_Normalise:
        CallBuild<_SIMDType, PerturbType::Gradient_Normalise>(noiseType, fractalType, args...);
        break;
    case PerturbType::GradientFractal_Normalise:
        CallBuild<_SIMDType, PerturbType::GradientFractal_Normalise>(noiseType, fractalType, args...);
        break;
    }
}


//#define SET_MAP_BUILDER(f)
template<SIMDType _SIMDType>
void BuildMap(const PerturbValues<_SIMDType> &perturb, 
    float *noiseSet, float *xMap, float *yMap, float *zMap, int xSize, int ySize, int zSize, float scaleModifier)
{
//    typename SIMD<_SIMDType>::Int seedV=SIMD<_SIMDType>::set(m_seed);
//    typename SIMD<_SIMDType>::Float FreqV=SIMD<_SIMDType>::set(m_frequency);
//
//	int index = 0; 
//	int maxIndex = xSize * ySize * zSize; 
//	
//	for (; index < maxIndex ; index += SIMD<_SIMDType>::vectorSize())
//	{
//		typename SIMD<_SIMDType>::Float xF = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::load(&xMap[index]), FreqV);
//		typename SIMD<_SIMDType>::Float yF = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::load(&yMap[index]), FreqV);
//		typename SIMD<_SIMDType>::Float zF = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::load(&zMap[index]), FreqV);
//		
//        Perturb<_SIMDType, _PerturbType>::_<_SIMDType, _PerturbType>::_(seedV, perturb, xF, yF, zF);
//		typename SIMD<_SIMDType>::Float result=GetSingle<_SIMDType, _NoiseType, _FractalType>::_(seedV, xF, yF, zF);
//		SIMD<_SIMDType>::store(&noiseSet[index], result);
//	}
}


//#define FILL_SET(func) 
template<SIMDType _SIMDType>
void NoiseSIMD<_SIMDType>::FillSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
{
	assert(noiseSet);
	SIMD<_SIMDType>::zeroAll();

    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails, scaleModifier);
    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
	
//    Build<_SIMDType, _NoiseType, FractalType::None, _PerturbType>::_(seedV, perturb, noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, xFreqV, yFreqV, zFreqV);
    CallBuild<_SIMDType>(m_noiseType, m_fractalType, m_perturbType,
        noise, perturb, noiseSet, xStart, yStart, zStart, xSize, ySize, zSize);
	
	SIMD<_SIMDType>::zeroAll();
}

//#define FILL_SET_MAP(func)
template<SIMDType _SIMDType>
void NoiseSIMD<_SIMDType>::FillSetMap(float* noiseSet, float* xMap, float* yMap, float* zMap, int xSize, int ySize, int zSize)
{
	assert(noiseSet);
    SIMD<_SIMDType>::zeroAll();

    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails);
    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
	
//    BuildMap<_SIMDType, _NoiseType, FractalType::None, _PerturbType>::_(perturb, noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
	
	SIMD<_SIMDType>::zeroAll();
}

//#define FILL_FRACTAL_SET(func)
template<SIMDType _SIMDType>
void NoiseSIMD<_SIMDType>::FillFractalSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
{
	assert(noiseSet);
	SIMD<_SIMDType>::zeroAll();
	
    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails, scaleModifier);
    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
	
    CallBuild<_SIMDType>(m_noiseType, m_fractalType, m_perturbType,
        noise, perturb, noiseSet, xStart, yStart, zStart, xSize, ySize, zSize);
	SIMD<_SIMDType>::zeroAll();
}

//#define FILL_FRACTAL_SET_MAP(func)
template<SIMDType _SIMDType>
void NoiseSIMD<_SIMDType>::FillFractalSetMap(float* noiseSet, float* xMap, float* yMap, float* zMap, int xSize, int ySize, int zSize)
{
	assert(noiseSet);
	SIMD<_SIMDType>::zeroAll();
	
    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails);
    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
	
//	BuildMap<_SIMDType, _NoiseType, _FractalType, _PerturbType>(perturb, noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, scaleModifier);
	SIMD<_SIMDType>::zeroAll();
}

#ifdef FN_ALIGNED_SETS
#define SIZE_MASK
#else
#define SIZE_MASK & ~(SIMD<_SIMDType>::vectorSize() - 1)
#endif

//#define VECTOR_SET_BUILDER(f)
template<SIMDType _SIMDType, NoiseType _NoiseType, FractalType _FractalType, PerturbType _PerturbType>
void BuildVectorSet(const NoiseValues<_SIMDType> &noise, const PerturbValues<_SIMDType> &perturb, float* noiseSet, FastNoiseVectorSet* vectorSet,
    const typename SIMD<_SIMDType>::Float &xOffsetV, const typename SIMD<_SIMDType>::Float &yOffsetV, const typename SIMD<_SIMDType>::Float &zOffsetV)
{
    int index=0;
    int loopMax=vectorSet->size SIZE_MASK;

    while(index<loopMax)
    {
        typename SIMD<_SIMDType>::Float xF=SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::load(&vectorSet->xSet[index]), noise.xFreqV, xOffsetV); 
        typename SIMD<_SIMDType>::Float yF=SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::load(&vectorSet->ySet[index]), noise.yFreqV, yOffsetV); 
        typename SIMD<_SIMDType>::Float zF=SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::load(&vectorSet->zSet[index]), noise.zFreqV, zOffsetV); 
        
        Perturb<_SIMDType, _PerturbType>::_(noise.seedV, perturb, xF, yF, zF);
        typename SIMD<_SIMDType>::Float result=GetSingle<_SIMDType, _NoiseType, _FractalType>::_(noise, xF, yF, zF);
        SIMD<_SIMDType>::store(&noiseSet[index], result); 
        index+=SIMD<_SIMDType>::vectorSize(); 
    }
    
#ifndef FN_ALIGNED_SETS
    if(loopMax!=vectorSet->size)
    {
        std::size_t remaining=(vectorSet->size-loopMax)*4; 
        
        typename SIMD<_SIMDType>::Float xF=SIMD<_SIMDType>::load(&vectorSet->xSet[loopMax]); 
        typename SIMD<_SIMDType>::Float yF=SIMD<_SIMDType>::load(&vectorSet->ySet[loopMax]); 
        typename SIMD<_SIMDType>::Float zF=SIMD<_SIMDType>::load(&vectorSet->zSet[loopMax]); 
        
        xF=SIMD<_SIMDType>::mulAdd(xF, noise.xFreqV, xOffsetV); 
        yF=SIMD<_SIMDType>::mulAdd(yF, noise.yFreqV, yOffsetV); 
        zF=SIMD<_SIMDType>::mulAdd(zF, noise.zFreqV, zOffsetV); 
        
        typename SIMD<_SIMDType>::Float result=GetSingle<_SIMDType, _NoiseType, _FractalType>::_(noise, xF, yF, zF);
        std::memcpy(&noiseSet[index], &result, remaining); 
    }
#endif
}

template<SIMDType _SIMDType, FractalType _FractalType, PerturbType _PerturbType, typename... _Types>
static void CallBuildVectorSet(NoiseType noiseType, _Types... args)
{
    switch(noiseType)
    {
    case NoiseType::Value:
        BuildVectorSet<_SIMDType, NoiseType::Value, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::ValueFractal:
        BuildVectorSet<_SIMDType, NoiseType::ValueFractal, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Perlin:
        BuildVectorSet<_SIMDType, NoiseType::Perlin, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::PerlinFractal:
        BuildVectorSet<_SIMDType, NoiseType::PerlinFractal, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Simplex:
        BuildVectorSet<_SIMDType, NoiseType::Simplex, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::SimplexFractal:
        BuildVectorSet<_SIMDType, NoiseType::SimplexFractal, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::WhiteNoise:
        BuildVectorSet<_SIMDType, NoiseType::WhiteNoise, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Cellular:
        BuildVectorSet<_SIMDType, NoiseType::Cellular, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::Cubic:
        BuildVectorSet<_SIMDType, NoiseType::Cubic, _FractalType, _PerturbType>(args...);
        break;
    case NoiseType::CubicFractal:
        BuildVectorSet<_SIMDType, NoiseType::CubicFractal, _FractalType, _PerturbType>(args...);
        break;
    }
}

template<SIMDType _SIMDType, PerturbType _PerturbType, typename... _Types>
static void CallBuildVectorSet(NoiseType noiseType, FractalType fractalType, _Types... args)
{
    switch(fractalType)
    {
    case FractalType::None:
        CallBuildVectorSet<_SIMDType, FractalType::None, _PerturbType>(noiseType, args...);
        break;
    case FractalType::FBM:
        CallBuildVectorSet<_SIMDType, FractalType::FBM, _PerturbType>(noiseType, args...);
        break;
    case FractalType::Billow:
        CallBuildVectorSet<_SIMDType, FractalType::Billow, _PerturbType>(noiseType, args...);
        break;
    case FractalType::RigidMulti:
        CallBuildVectorSet<_SIMDType, FractalType::RigidMulti, _PerturbType>(noiseType, args...);
        break;
    }
}

template<SIMDType _SIMDType, typename... _Types>
static void CallBuildVectorSet(NoiseType noiseType, FractalType fractalType, PerturbType perturbType, _Types... args)
{
    switch(perturbType)
    {
    case PerturbType::None:
        CallBuildVectorSet<_SIMDType, PerturbType::None>(noiseType, fractalType, args...);
        break;
    case PerturbType::Gradient:
        CallBuildVectorSet<_SIMDType, PerturbType::Gradient>(noiseType, fractalType, args...);
        break;
    case PerturbType::GradientFractal:
        CallBuildVectorSet<_SIMDType, PerturbType::GradientFractal>(noiseType, fractalType, args...);
        break;
    case PerturbType::Normalise:
        CallBuildVectorSet<_SIMDType, PerturbType::Normalise>(noiseType, fractalType, args...);
        break;
    case PerturbType::Gradient_Normalise:
        CallBuildVectorSet<_SIMDType, PerturbType::Gradient_Normalise>(noiseType, fractalType, args...);
        break;
    case PerturbType::GradientFractal_Normalise:
        CallBuildVectorSet<_SIMDType, PerturbType::GradientFractal_Normalise>(noiseType, fractalType, args...);
        break;
    }
}

//#define FILL_VECTOR_SET(func)
template<SIMDType _SIMDType>
void NoiseSIMD<_SIMDType>::FillSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)
{
	assert(noiseSet);
	assert(vectorSet);
	assert(vectorSet->size >= 0);
	SIMD<_SIMDType>::zeroAll();
	
    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails);
	typename SIMD<_SIMDType>::Float xOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(xOffset), noise.xFreqV);
	typename SIMD<_SIMDType>::Float yOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(yOffset), noise.yFreqV);
	typename SIMD<_SIMDType>::Float zOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(zOffset), noise.zFreqV);
    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
	
    CallBuildVectorSet<_SIMDType>(m_noiseType, m_fractalType, m_perturbType,
        noise, perturb, noiseSet, vectorSet, xOffsetV, yOffsetV, zOffsetV);
	SIMD<_SIMDType>::zeroAll();
}

//#define FILL_FRACTAL_VECTOR_SET(func)
template<SIMDType _SIMDType>
void NoiseSIMD<_SIMDType>::FillFractalSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)
{
	assert(noiseSet);
	assert(vectorSet);
	assert(vectorSet->size >= 0);
	SIMD<_SIMDType>::zeroAll();
	
    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails);
	typename SIMD<_SIMDType>::Float xOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(xOffset), noise.xFreqV);
	typename SIMD<_SIMDType>::Float yOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(yOffset), noise.yFreqV);
	typename SIMD<_SIMDType>::Float zOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(zOffset), noise.zFreqV);

    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
	
    CallBuildVectorSet<_SIMDType>(m_noiseType, m_fractalType, m_perturbType,
        noise, perturb, noiseSet, vectorSet, xOffsetV, yOffsetV, zOffsetV);
	SIMD<_SIMDType>::zeroAll();
}

template<SIMDType _SIMDType>
void NoiseSIMD<_SIMDType>::FillWhiteNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;

	assert(noiseSet);
	SIMD<_SIMDType>::zeroAll();
	typename SIMD<_SIMDType>::Int seedV = SIMD<_SIMDType>::set(m_noiseDetails.seed);

	if ((zSize & (SIMD<_SIMDType>::vectorSize() - 1)) == 0)
	{
		typename SIMD<_SIMDType>::Int x = SIMD<_SIMDType>::mul(SIMD<_SIMDType>::set(xStart), Constant::numi_xPrime);
		typename SIMD<_SIMDType>::Int yBase = SIMD<_SIMDType>::mul(SIMD<_SIMDType>::set(yStart), Constant::numi_yPrime);
		typename SIMD<_SIMDType>::Int zBase = SIMD<_SIMDType>::mul(SIMD<_SIMDType>::add(Constant::numi_incremental, SIMD<_SIMDType>::set(zStart)), Constant::numi_zPrime);

		typename SIMD<_SIMDType>::Int zStep = SIMD<_SIMDType>::mul(Constant::numi_vectorSize, Constant::numi_zPrime);

		int index = 0;

		for (int ix = 0; ix < xSize; ix++)
		{
			typename SIMD<_SIMDType>::Int y = yBase;

			for (int iy = 0; iy < ySize; iy++)
			{
				typename SIMD<_SIMDType>::Int z = zBase;

				SIMD<_SIMDType>::store(&noiseSet[index], ValCoord<_SIMDType>(seedV, x, y, z));

				int iz = SIMD<_SIMDType>::vectorSize();
				while (iz < zSize)
				{
					z = SIMD<_SIMDType>::add(z, zStep);
					index += SIMD<_SIMDType>::vectorSize();
					iz += SIMD<_SIMDType>::vectorSize();

					SIMD<_SIMDType>::store(&noiseSet[index], ValCoord<_SIMDType>(seedV, x, y, z));
				}
				index += SIMD<_SIMDType>::vectorSize();
				y = SIMD<_SIMDType>::add(y, Constant::numi_yPrime);
			}
			x = SIMD<_SIMDType>::add(x, Constant::numi_xPrime);
		}
	}
	else
	{
		typename SIMD<_SIMDType>::Int ySizeV = SIMD<_SIMDType>::set(ySize);
		typename SIMD<_SIMDType>::Int zSizeV = SIMD<_SIMDType>::set(zSize);

		typename SIMD<_SIMDType>::Int yEndV = SIMD<_SIMDType>::set(yStart + ySize - 1);
		typename SIMD<_SIMDType>::Int zEndV = SIMD<_SIMDType>::set(zStart + zSize - 1);

		typename SIMD<_SIMDType>::Int x = SIMD<_SIMDType>::set(xStart);
		typename SIMD<_SIMDType>::Int y = SIMD<_SIMDType>::set(yStart);
		typename SIMD<_SIMDType>::Int z = SIMD<_SIMDType>::add(SIMD<_SIMDType>::set(zStart), Constant::numi_incremental);
//		AXIS_RESET(zSize, 1);
        axisReset<_SIMDType>(x, y, z, ySizeV, yEndV, zSizeV, zEndV, zSize, 1);

		int index = 0;
		int maxIndex = xSize * ySize * zSize;

		for (; index < maxIndex - SIMD<_SIMDType>::vectorSize(); index += SIMD<_SIMDType>::vectorSize())
		{
			SIMD<_SIMDType>::store(&noiseSet[index], ValCoord<_SIMDType>(seedV, SIMD<_SIMDType>::mul(x, Constant::numi_xPrime), SIMD<_SIMDType>::mul(y, Constant::numi_yPrime), SIMD<_SIMDType>::mul(z, Constant::numi_zPrime)));

			z = SIMD<_SIMDType>::add(z, Constant::numi_vectorSize);

//			AXIS_RESET(zSize, 0);
            axisReset<_SIMDType>(x, y, z, ySizeV, yEndV, zSizeV, zEndV, zSize, 0);
		}
		typename SIMD<_SIMDType>::Float result = ValCoord<_SIMDType>(seedV, SIMD<_SIMDType>::mul(x, Constant::numi_xPrime), SIMD<_SIMDType>::mul(y, Constant::numi_yPrime), SIMD<_SIMDType>::mul(z, Constant::numi_zPrime));
		STORE_LAST_RESULT(&noiseSet[index], result);
	}
	SIMD<_SIMDType>::zeroAll();
}

//#define Euclidean_DISTANCE(_x, _y, _z) SIMD<_SIMDType>::mulAdd(_x, _x, SIMD<_SIMDType>::mulAdd(_y, _y, SIMD<_SIMDType>::mul(_z, _z)))
//#define Manhattan_DISTANCE(_x, _y, _z) SIMD<_SIMDType>::add(SIMD<_SIMDType>::add(SIMD<_SIMDType>::abs(_x), SIMD<_SIMDType>::abs(_y)), SIMD<_SIMDType>::abs(_z))
//#define Natural_DISTANCE(_x, _y, _z) SIMD<_SIMDType>::add(Euclidean_DISTANCE(_x,_y,_z), Manhattan_DISTANCE(_x,_y,_z))

template<SIMDType _SIMDType, CellularDistance _CellularDistance>
struct Distance
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &_x, const typename SIMD<_SIMDType>::Float &_y, const typename SIMD<_SIMDType>::Float &_z)
    {//Euclidean
        return SIMD<_SIMDType>::mulAdd(_x, _x, SIMD<_SIMDType>::mulAdd(_y, _y, SIMD<_SIMDType>::mulf(_z, _z)));
    }
};

template<SIMDType _SIMDType>
struct Distance<_SIMDType, CellularDistance::Manhattan>
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &_x, const typename SIMD<_SIMDType>::Float &_y, const typename SIMD<_SIMDType>::Float &_z)
    {
        return SIMD<_SIMDType>::add(SIMD<_SIMDType>::add(SIMD<_SIMDType>::abs(_x), SIMD<_SIMDType>::abs(_y)), SIMD<_SIMDType>::abs(_z));
    }
};

template<SIMDType _SIMDType>
struct Distance<_SIMDType, CellularDistance::Natural>
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &_x, const typename SIMD<_SIMDType>::Float &_y, const typename SIMD<_SIMDType>::Float &_z)
    {
        return SIMD<_SIMDType>::add(Distance<_SIMDType, CellularDistance::Euclidean>::_(_x, _y, _z), Distance<_SIMDType, CellularDistance::Manhattan>::_(_x, _y, _z));
    }
};

//#define Distance2_RETURN(_distance, _distance2) (_distance2)
//#define Distance2Add_RETURN(_distance, _distance2) SIMD<_SIMDType>::add(_distance, _distance2)
//#define Distance2Sub_RETURN(_distance, _distance2) SIMD<_SIMDType>::sub(_distance2, _distance)
//#define Distance2Mul_RETURN(_distance, _distance2) SIMD<_SIMDType>::mul(_distance, _distance2)
//#define Distance2Div_RETURN(_distance, _distance2) SIMD<_SIMDType>::div(_distance, _distance2)

template<SIMDType _SIMDType, CellularReturnType _CellularReturnType>
struct ReturnDistance
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &distance, const typename SIMD<_SIMDType>::Float &distance2)
    {
        return distance;
    }
};

template<SIMDType _SIMDType>
struct ReturnDistance<_SIMDType, CellularReturnType::Distance2>
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &distance, const typename SIMD<_SIMDType>::Float &distance2)
    {
        return distance2;
    }
};

template<SIMDType _SIMDType>
struct ReturnDistance<_SIMDType, CellularReturnType::Distance2Add>
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &distance, const typename SIMD<_SIMDType>::Float &distance2)
    {
        return SIMD<_SIMDType>::add(distance, distance2);
    }
};

template<SIMDType _SIMDType>
struct ReturnDistance<_SIMDType, CellularReturnType::Distance2Sub>
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &distance, const typename SIMD<_SIMDType>::Float &distance2)
    {
        return SIMD<_SIMDType>::sub(distance2, distance);
    }
};

template<SIMDType _SIMDType>
struct ReturnDistance<_SIMDType, CellularReturnType::Distance2Mul>
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &distance, const typename SIMD<_SIMDType>::Float &distance2)
    {
        return SIMD<_SIMDType>::mulf(distance, distance2);
    }
};

template<SIMDType _SIMDType>
struct ReturnDistance<_SIMDType, CellularReturnType::Distance2Div>
{
    static typename SIMD<_SIMDType>::Float _(const typename SIMD<_SIMDType>::Float &distance, const typename SIMD<_SIMDType>::Float &distance2)
    {
        return SIMD<_SIMDType>::div(distance, distance2);
    }
};

//#define CELLULAR_VALUE_SINGLE(distanceFunc)
template<SIMDType _SIMDType, CellularDistance _CellularDistance>
static typename SIMD<_SIMDType>::Float VECTORCALL CellularValueSingle(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z, typename SIMD<_SIMDType>::Float cellJitter)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
	typename SIMD<_SIMDType>::Float distance = Constant::numf_999999;
	typename SIMD<_SIMDType>::Float cellValue = SIMD<_SIMDType>::undefinedFloat();
	
	typename SIMD<_SIMDType>::Int xc     = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(x), Constant::numi_1);
	typename SIMD<_SIMDType>::Int ycBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(y), Constant::numi_1);
	typename SIMD<_SIMDType>::Int zcBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(z), Constant::numi_1);
	
	typename SIMD<_SIMDType>::Float xcf     = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(xc), x);
	typename SIMD<_SIMDType>::Float ycfBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(ycBase), y);
	typename SIMD<_SIMDType>::Float zcfBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(zcBase), z);
	
	xc     = SIMD<_SIMDType>::mul(xc, Constant::numi_xPrime);
	ycBase = SIMD<_SIMDType>::mul(ycBase, Constant::numi_yPrime);
	zcBase = SIMD<_SIMDType>::mul(zcBase, Constant::numi_zPrime);
	
	for (int xi = 0; xi < 3; xi++)
	{
		typename SIMD<_SIMDType>::Float ycf = ycfBase;
		typename SIMD<_SIMDType>::Int yc = ycBase;
		for (int yi = 0; yi < 3; yi++)
		{
			typename SIMD<_SIMDType>::Float zcf = zcfBase;
			typename SIMD<_SIMDType>::Int zc = zcBase;
			for (int zi = 0; zi < 3; zi++)
			{
				typename SIMD<_SIMDType>::Int hash = HashHB<_SIMDType>(seed, xc, yc, zc);
				typename SIMD<_SIMDType>::Float xd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(hash, Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float yd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,10), Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float zd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,20), Constant::numi_bit10Mask)), Constant::numf_511_5);
				
				typename SIMD<_SIMDType>::Float invMag = SIMD<_SIMDType>::mulf(cellJitter, SIMD<_SIMDType>::invSqrt(SIMD<_SIMDType>::mulAdd(xd, xd, SIMD<_SIMDType>::mulAdd(yd, yd, SIMD<_SIMDType>::mulf(zd, zd)))));
				
				xd = SIMD<_SIMDType>::mulAdd(xd, invMag, xcf);
				yd = SIMD<_SIMDType>::mulAdd(yd, invMag, ycf);
				zd = SIMD<_SIMDType>::mulAdd(zd, invMag, zcf);
				
				typename SIMD<_SIMDType>::Float newCellValue = SIMD<_SIMDType>::mulf(Constant::numf_hash2Float, SIMD<_SIMDType>::convert(hash));
				typename SIMD<_SIMDType>::Float newDistance = Distance<_SIMDType, _CellularDistance>(xd, yd, zd);
				
				typename SIMD<_SIMDType>::Mask closer = SIMD<_SIMDType>::lessThan(newDistance, distance);
				
				distance = SIMD<_SIMDType>::min(newDistance, distance);
				cellValue = SIMD<_SIMDType>::blend(cellValue, newCellValue, closer);
				
				zcf = SIMD<_SIMDType>::add(zcf, Constant::numf_1);
				zc = SIMD<_SIMDType>::add(zc, Constant::numi_zPrime);
			}
			ycf = SIMD<_SIMDType>::add(ycf, Constant::numf_1);
			yc = SIMD<_SIMDType>::add(yc, Constant::numi_yPrime);
		}
		xcf = SIMD<_SIMDType>::add(xcf, Constant::numf_1);
		xc = SIMD<_SIMDType>::add(xc, Constant::numi_xPrime);
	}
	
	return cellValue;
}

template<SIMDType _SIMDType>
struct NoiseLookupSettings
{
	NoiseType type;
	typename SIMD<_SIMDType>::Float frequency;
	FractalType fractalType;
	int fractalOctaves;
	typename SIMD<_SIMDType>::Float fractalLacunarity;
	typename SIMD<_SIMDType>::Float fractalGain;
	typename SIMD<_SIMDType>::Float fractalBounding;
};

//#define CELLULAR_LOOKUP_FRACTAL_VALUE(noiseType){
template<SIMDType _SIMDType, NoiseType _NoiseType, FractalType _FractalType>
typename SIMD<_SIMDType>::Float FractalCelluarLookup(const typename SIMD<_SIMDType>::Int &seedV, const typename SIMD<_SIMDType>::Float &xF, const typename SIMD<_SIMDType>::Float &yF, const typename SIMD<_SIMDType>::Float &zF, const NoiseLookupSettings<_SIMDType>& noiseLookupSettings)
{
    typename SIMD<_SIMDType>::Float lacunarityV = noiseLookupSettings.fractalLacunarity;
    typename SIMD<_SIMDType>::Float gainV = noiseLookupSettings.fractalGain;
    typename SIMD<_SIMDType>::Float fractalBoundingV = noiseLookupSettings.fractalBounding;
    int m_octaves = noiseLookupSettings.fractalOctaves;

    return GetSingle<_SIMDType, _NoiseType, _FractalType>::_(seedV, xF, yF, zF);
}

//#define CELLULAR_LOOKUP_SINGLE(distanceFunc)
template<SIMDType _SIMDType, NoiseType _NoiseType, FractalType _FractalType, PerturbType _PerturbType, CellularDistance _CellularDistance>
static typename SIMD<_SIMDType>::Float VECTORCALL CellularLookupSingle(typename SIMD<_SIMDType>::Int seedV, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z, typename SIMD<_SIMDType>::Float cellJitter, const NoiseLookupSettings<_SIMDType>& noiseLookupSettings)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
	typename SIMD<_SIMDType>::Float distance = Constant::numf_999999;
	typename SIMD<_SIMDType>::Float xCell = SIMD<_SIMDType>::undefinedFloat();
	typename SIMD<_SIMDType>::Float yCell = SIMD<_SIMDType>::undefinedFloat();
	typename SIMD<_SIMDType>::Float zCell = SIMD<_SIMDType>::undefinedFloat();
	
	typename SIMD<_SIMDType>::Int xc     = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(x), Constant::numi_1);
	typename SIMD<_SIMDType>::Int ycBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(y), Constant::numi_1);
	typename SIMD<_SIMDType>::Int zcBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(z), Constant::numi_1);
	
	typename SIMD<_SIMDType>::Float xcf     = SIMD<_SIMDType>::convert(xc);
	typename SIMD<_SIMDType>::Float ycfBase = SIMD<_SIMDType>::convert(ycBase);
	typename SIMD<_SIMDType>::Float zcfBase = SIMD<_SIMDType>::convert(zcBase);
	
	xc     = SIMD<_SIMDType>::mul(xc, Constant::numi_xPrime);
	ycBase = SIMD<_SIMDType>::mul(ycBase, Constant::numi_yPrime);
	zcBase = SIMD<_SIMDType>::mul(zcBase, Constant::numi_zPrime);
	
	for (int xi = 0; xi < 3; xi++)
	{
		typename SIMD<_SIMDType>::Float ycf = ycfBase;
		typename SIMD<_SIMDType>::Int yc = ycBase;
		typename SIMD<_SIMDType>::Float xLocal = SIMD<_SIMDType>::sub(xcf, x);
		for (int yi = 0; yi < 3; yi++)
		{
			typename SIMD<_SIMDType>::Float zcf = zcfBase;
			typename SIMD<_SIMDType>::Int zc = zcBase;
			typename SIMD<_SIMDType>::Float yLocal = SIMD<_SIMDType>::sub(ycf, y);
			for (int zi = 0; zi < 3; zi++)
			{
				typename SIMD<_SIMDType>::Float zLocal = SIMD<_SIMDType>::sub(zcf, z);
				
				typename SIMD<_SIMDType>::Int hash = HashHB<_SIMDType>(seedV, xc, yc, zc);
				typename SIMD<_SIMDType>::Float xd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(hash, Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float yd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,10), Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float zd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,20), Constant::numi_bit10Mask)), Constant::numf_511_5);
				
				typename SIMD<_SIMDType>::Float invMag = SIMD<_SIMDType>::mulf(cellJitter, SIMD<_SIMDType>::invSqrt(SIMD<_SIMDType>::mulAdd(xd, xd, SIMD<_SIMDType>::mulAdd(yd, yd, SIMD<_SIMDType>::mulf(zd, zd)))));
				
				typename SIMD<_SIMDType>::Float xCellNew = SIMD<_SIMDType>::mulf(xd, invMag);
				typename SIMD<_SIMDType>::Float yCellNew = SIMD<_SIMDType>::mulf(yd, invMag);
				typename SIMD<_SIMDType>::Float zCellNew = SIMD<_SIMDType>::mulf(zd, invMag);
				
				xd = SIMD<_SIMDType>::add(xCellNew, xLocal);
				yd = SIMD<_SIMDType>::add(yCellNew, yLocal);
				zd = SIMD<_SIMDType>::add(zCellNew, zLocal);
				
				xCellNew = SIMD<_SIMDType>::add(xCellNew, xcf); 
				yCellNew = SIMD<_SIMDType>::add(yCellNew, ycf); 
				zCellNew = SIMD<_SIMDType>::add(zCellNew, zcf); 
				
				typename SIMD<_SIMDType>::Float newDistance = Distance<_SIMDType, _CellularDistance>(xd, yd, zd);
				
				typename SIMD<_SIMDType>::Mask closer = SIMD<_SIMDType>::lessThan(newDistance, distance);
				
				distance = SIMD<_SIMDType>::min(newDistance, distance);
				xCell = SIMD<_SIMDType>::blend(xCell, xCellNew, closer);
				yCell = SIMD<_SIMDType>::blend(yCell, yCellNew, closer);
				zCell = SIMD<_SIMDType>::blend(zCell, zCellNew, closer);
				
				zcf = SIMD<_SIMDType>::add(zcf, Constant::numf_1);
				zc = SIMD<_SIMDType>::add(zc, Constant::numi_zPrime);
			}
			ycf = SIMD<_SIMDType>::add(ycf, Constant::numf_1);
			yc = SIMD<_SIMDType>::add(yc, Constant::numi_yPrime);
		}
		xcf = SIMD<_SIMDType>::add(xcf, Constant::numf_1);
		xc = SIMD<_SIMDType>::add(xc, Constant::numi_xPrime);
	}
	
	typename SIMD<_SIMDType>::Float xF = SIMD<_SIMDType>::mulf(xCell, noiseLookupSettings.frequency);
	typename SIMD<_SIMDType>::Float yF = SIMD<_SIMDType>::mulf(yCell, noiseLookupSettings.frequency);
	typename SIMD<_SIMDType>::Float zF = SIMD<_SIMDType>::mulf(zCell, noiseLookupSettings.frequency);
	typename SIMD<_SIMDType>::Float result=FractalCelluarLookup<_SIMDType, _NoiseType, _FractalType>(seedV, xF, yF, zF, noiseLookupSettings);

	return result;
}

//#define CELLULAR_DISTANCE_SINGLE(distanceFunc)
template<SIMDType _SIMDType, NoiseType _NoiseType, FractalType _FractalType, PerturbType _PerturbType, CellularDistance _CellularDistance>
static typename SIMD<_SIMDType>::Float VECTORCALL CellularDistanceSingle(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z, typename SIMD<_SIMDType>::Float cellJitter)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
	typename SIMD<_SIMDType>::Float distance = Constant::numf_999999;
	
	typename SIMD<_SIMDType>::Int xc     = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(x), Constant::numi_1);
	typename SIMD<_SIMDType>::Int ycBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(y), Constant::numi_1);
	typename SIMD<_SIMDType>::Int zcBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(z), Constant::numi_1);
	
	typename SIMD<_SIMDType>::Float xcf     = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(xc), x);
	typename SIMD<_SIMDType>::Float ycfBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(ycBase), y);
	typename SIMD<_SIMDType>::Float zcfBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(zcBase), z);
	
	xc     = SIMD<_SIMDType>::mul(xc, Constant::numi_xPrime);
	ycBase = SIMD<_SIMDType>::mul(ycBase, Constant::numi_yPrime);
	zcBase = SIMD<_SIMDType>::mul(zcBase, Constant::numi_zPrime);
	
	for (int xi = 0; xi < 3; xi++)
	{
		typename SIMD<_SIMDType>::Float ycf = ycfBase;
		typename SIMD<_SIMDType>::Int yc = ycBase;
		for (int yi = 0; yi < 3; yi++)
		{
			typename SIMD<_SIMDType>::Float zcf = zcfBase;
			typename SIMD<_SIMDType>::Int zc = zcBase;
			for (int zi = 0; zi < 3; zi++)
			{
				typename SIMD<_SIMDType>::Int hash = HashHB<_SIMDType>(seed, xc, yc, zc);
				typename SIMD<_SIMDType>::Float xd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(hash, Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float yd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,10), Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float zd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,20), Constant::numi_bit10Mask)), Constant::numf_511_5);
				
				typename SIMD<_SIMDType>::Float invMag = SIMD<_SIMDType>::mulf(cellJitter, SIMD<_SIMDType>::invSqrt(SIMD<_SIMDType>::mulAdd(xd, xd, SIMD<_SIMDType>::mulAdd(yd, yd, SIMD<_SIMDType>::mulf(zd, zd)))));
				
				xd = SIMD<_SIMDType>::mulAdd(xd, invMag, xcf);
				yd = SIMD<_SIMDType>::mulAdd(yd, invMag, ycf);
				zd = SIMD<_SIMDType>::mulAdd(zd, invMag, zcf);
				
				typename SIMD<_SIMDType>::Float newDistance = Distance<_SIMDType, _CellularDistance>::_(xd, yd, zd);
				
				distance = SIMD<_SIMDType>::min(distance, newDistance);
				
				zcf = SIMD<_SIMDType>::add(zcf, Constant::numf_1);
				zc = SIMD<_SIMDType>::add(zc, Constant::numi_zPrime);
			}
			ycf = SIMD<_SIMDType>::add(ycf, Constant::numf_1);
			yc = SIMD<_SIMDType>::add(yc, Constant::numi_yPrime);
		}
		xcf = SIMD<_SIMDType>::add(xcf, Constant::numf_1);
		xc = SIMD<_SIMDType>::add(xc, Constant::numi_xPrime);
	}
	
	return distance;
}

//#define CELLULAR_DISTANCE2_SINGLE(distanceFunc, returnFunc)
template<SIMDType _SIMDType, CellularDistance _CellularDistance, CellularReturnType _CellularReturnType>
static typename SIMD<_SIMDType>::Float VECTORCALL CellularReturnDistanceSingle(typename SIMD<_SIMDType>::Int seed, typename SIMD<_SIMDType>::Float x, typename SIMD<_SIMDType>::Float y, typename SIMD<_SIMDType>::Float z, typename SIMD<_SIMDType>::Float cellJitter, int index0, int index1)
{
    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
	typename SIMD<_SIMDType>::Float distance[FN_CELLULAR_INDEX_MAX+1] = {Constant::numf_999999,Constant::numf_999999,Constant::numf_999999,Constant::numf_999999};
	
	typename SIMD<_SIMDType>::Int xc     = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(x), Constant::numi_1);
	typename SIMD<_SIMDType>::Int ycBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(y), Constant::numi_1);
	typename SIMD<_SIMDType>::Int zcBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(z), Constant::numi_1);
	
	typename SIMD<_SIMDType>::Float xcf     = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(xc), x);
	typename SIMD<_SIMDType>::Float ycfBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(ycBase), y);
	typename SIMD<_SIMDType>::Float zcfBase = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(zcBase), z);
	
	xc     = SIMD<_SIMDType>::mul(xc, Constant::numi_xPrime);
	ycBase = SIMD<_SIMDType>::mul(ycBase, Constant::numi_yPrime);
	zcBase = SIMD<_SIMDType>::mul(zcBase, Constant::numi_zPrime);
	
	for (int xi = 0; xi < 3; xi++)
	{
		typename SIMD<_SIMDType>::Float ycf = ycfBase;
		typename SIMD<_SIMDType>::Int yc = ycBase;
		for (int yi = 0; yi < 3; yi++)
		{
			typename SIMD<_SIMDType>::Float zcf = zcfBase;
			typename SIMD<_SIMDType>::Int zc = zcBase;
			for (int zi = 0; zi < 3; zi++)
			{
				typename SIMD<_SIMDType>::Int hash = HashHB<_SIMDType>(seed, xc, yc, zc);
				typename SIMD<_SIMDType>::Float xd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(hash, Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float yd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,10), Constant::numi_bit10Mask)), Constant::numf_511_5);
				typename SIMD<_SIMDType>::Float zd = SIMD<_SIMDType>::sub(SIMD<_SIMDType>::convert(SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(hash,20), Constant::numi_bit10Mask)), Constant::numf_511_5);
				
				typename SIMD<_SIMDType>::Float invMag = SIMD<_SIMDType>::mulf(cellJitter, SIMD<_SIMDType>::invSqrt(SIMD<_SIMDType>::mulAdd(xd, xd, SIMD<_SIMDType>::mulAdd(yd, yd, SIMD<_SIMDType>::mulf(zd, zd)))));
				
				xd = SIMD<_SIMDType>::mulAdd(xd, invMag, xcf);
				yd = SIMD<_SIMDType>::mulAdd(yd, invMag, ycf);
				zd = SIMD<_SIMDType>::mulAdd(zd, invMag, zcf);
				
				typename SIMD<_SIMDType>::Float newDistance = Distance<_SIMDType, _CellularDistance>::_(xd, yd, zd);
				
				for(int i = index1; i > 0; i--)
					distance[i] = SIMD<_SIMDType>::max(SIMD<_SIMDType>::min(distance[i], newDistance), distance[i-1]);
				distance[0] = SIMD<_SIMDType>::min(distance[0], newDistance);
				
				zcf = SIMD<_SIMDType>::add(zcf, Constant::numf_1);
				zc = SIMD<_SIMDType>::add(zc, Constant::numi_zPrime);
			}
			ycf = SIMD<_SIMDType>::add(ycf, Constant::numf_1);
			yc = SIMD<_SIMDType>::add(yc, Constant::numi_yPrime);
		}
		xcf = SIMD<_SIMDType>::add(xcf, Constant::numf_1);
		xc = SIMD<_SIMDType>::add(xc, Constant::numi_xPrime);
	}
	
	return ReturnDistance<_SIMDType, _CellularReturnType>::_(distance[index0], distance[index1]);
}

//template<SIMDType _SIMDType>
//void NoiseSIMD<_SIMDType>::FillCellularSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float scaleModifier)
//{
//	assert(noiseSet);
//	SIMD<_SIMDType>::zeroAll();
//    
//    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails, scaleModifier);
//    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
//
//    CellularValue
////	NoiseLookupSettings nls;
////    CallBuildMap<_SIMDType, _NoiseType, _FractalType, _PerturbType, _CellularDistance, _CellularReturnType>(seedV, perturb, noiseSet, xMap, yMap, zMap, xSize, ySize, zSize, cellJitterV);
//
//	SIMD<_SIMDType>::zeroAll();
//}
//
//template<SIMDType _SIMDType>
//void NoiseSIMD<_SIMDType>::FillCellularSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)
//{
//	assert(noiseSet);
//	assert(vectorSet);
//	assert(vectorSet->size >= 0);
//	SIMD<_SIMDType>::zeroAll();
//
//    NoiseValues<_SIMDType> noise=initNoise<_SIMDType>(m_noiseDetails);
//
//    typename SIMD<_SIMDType>::Float xOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(xOffset), noise.xFreqV);
//	typename SIMD<_SIMDType>::Float yOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(yOffset), noise.yFreqV);
//	typename SIMD<_SIMDType>::Float zOffsetV = SIMD<_SIMDType>::mulf(SIMD<_SIMDType>::set(zOffset), noise.zFreqV);
//	
//    PerturbValues<_SIMDType> perturb=initPerturb<_SIMDType>(m_perturbType, m_noiseDetails, m_perturbDetails);
//
//	int index = 0;
//	int loopMax = vectorSet->size SIZE_MASK;
////	NoiseLookupSettings nls;
//
//    CallBuildVectorSet<_SIMDType>(m_noiseType, m_fractalType, m_perturbType, m_cellularDistance, m_cellularReturnType,
//        noise, perturb, xOffsetV, yOffsetV, zOffsetV);
//
//	SIMD<_SIMDType>::zeroAll();
//}
//
//#define SAMPLE_INDEX(_x,_y,_z) ((_x) * yzSizeSample + (_y) * zSizeSample + (_z))
//#define SET_INDEX(_x,_y,_z) ((_x) * yzSize + (_y) * zSize + (_z))
//
//template<SIMDType _SIMDType>
//void NoiseSIMD<_SIMDType>::FillSampledNoiseSet(float* noiseSet, int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, int sampleScale)
//{
//    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
//
//	assert(noiseSet);
//	SIMD<_SIMDType>::zeroAll();
//
//	if (sampleScale <= 0)
//	{
//		FillSet(noiseSet, xStart, yStart, zStart, xSize, ySize, zSize, 1.0);
//		return;
//	}
//
//	int sampleSize = 1 << sampleScale;
//	int sampleMask = sampleSize - 1;
//	float scaleModifier = float(sampleSize);
//
//	int xOffset = (sampleSize - (xStart & sampleMask)) & sampleMask;
//	int yOffset = (sampleSize - (yStart & sampleMask)) & sampleMask;
//	int zOffset = (sampleSize - (zStart & sampleMask)) & sampleMask;
//
//	int xSizeSample = xSize + xOffset;
//	int ySizeSample = ySize + yOffset;
//	int zSizeSample = zSize + zOffset;
//
//	if (xSizeSample & sampleMask)
//		xSizeSample = (xSizeSample & ~sampleMask) + sampleSize;
//
//	if (ySizeSample & sampleMask)
//		ySizeSample = (ySizeSample & ~sampleMask) + sampleSize;
//
//	if (zSizeSample & sampleMask)
//		zSizeSample = (zSizeSample & ~sampleMask) + sampleSize;
//
//	xSizeSample = (xSizeSample >> sampleScale) + 1;
//	ySizeSample = (ySizeSample >> sampleScale) + 1;
//	zSizeSample = (zSizeSample >> sampleScale) + 1;
//
//	float* noiseSetSample = GetEmptySet(xSizeSample * ySizeSample * zSizeSample);
//	FillSet(noiseSetSample, xStart >> sampleScale, yStart >> sampleScale, zStart >> sampleScale, xSizeSample, ySizeSample, zSizeSample, scaleModifier);
//
//	int yzSizeSample = ySizeSample * zSizeSample;
//	int yzSize = ySize * zSize;
//
//	typename SIMD<_SIMDType>::Int axisMask = SIMD<_SIMDType>::set(sampleMask);
//	typename SIMD<_SIMDType>::Float axisScale = SIMD<_SIMDType>::set(1.f / scaleModifier);
//	typename SIMD<_SIMDType>::Float axisOffset = SIMD<_SIMDType>::mulf(axisScale, Constant::numf_0_5);
//
//	typename SIMD<_SIMDType>::Int sampleSizeSIMD = SIMD<_SIMDType>::set(sampleSize);
//	typename SIMD<_SIMDType>::Int xSIMD = SIMD<_SIMDType>::set(-xOffset);
//	typename SIMD<_SIMDType>::Int yBase = SIMD<_SIMDType>::set(-yOffset);
//	typename SIMD<_SIMDType>::Int zBase = SIMD<_SIMDType>::set(-zOffset);
//
//	int localCountMax = (1 << (sampleScale * 3));
//	int vMax = SIMD<_SIMDType>::vectorSize();
//
//#if SIMD_LEVEL == FN_NEON
//	typename SIMD<_SIMDType>::Int sampleScaleV = SIMD<_SIMDType>::set(-sampleScale);
//	typename SIMD<_SIMDType>::Int sampleScale2V = SIMD<_SIMDType>::mul(sampleScaleV, Constant::numi_2);
//#endif
//
//	for (int x = 0; x < xSizeSample - 1; x++)
//	{
//		typename SIMD<_SIMDType>::Int ySIMD = yBase;
//		for (int y = 0; y < ySizeSample - 1; y++)
//		{
//			typename SIMD<_SIMDType>::Int zSIMD = zBase;
//
//			typename SIMD<_SIMDType>::Float c001 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y, 0)]);
//			typename SIMD<_SIMDType>::Float c101 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y, 0)]);
//			typename SIMD<_SIMDType>::Float c011 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y + 1, 0)]);
//			typename SIMD<_SIMDType>::Float c111 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, 0)]);
//			for (int z = 0; z < zSizeSample - 1; z++)
//			{
//				typename SIMD<_SIMDType>::Float c000 = c001;
//				typename SIMD<_SIMDType>::Float c100 = c101;
//				typename SIMD<_SIMDType>::Float c010 = c011;
//				typename SIMD<_SIMDType>::Float c110 = c111;
//				c001 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y, z + 1)]);
//				c101 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y, z + 1)]);
//				c011 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y + 1, z + 1)]);
//				c111 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, z + 1)]);
//
//				typename SIMD<_SIMDType>::Int localCountSIMD = Constant::numi_incremental;
//
//				int localCount = 0;
//				while (localCount < localCountMax)
//				{
//                    uSIMD<SIMD<_SIMDType>::Int, SIMD<_SIMDType>::vectorSize()> xi, yi, zi;
//
//#if SIMD_LEVEL == FN_NEON
//					xi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftL(localCountSIMD, sampleScale2V), axisMask);
//					yi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftL(localCountSIMD, sampleScaleV), axisMask);
//#else
//					xi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(localCountSIMD, sampleScale * 2), axisMask);
//					yi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(localCountSIMD, sampleScale), axisMask);
//#endif
//
//					zi.m = SIMD<_SIMDType>::_and(localCountSIMD, axisMask);
//
//					typename SIMD<_SIMDType>::Float xf = SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::convert(xi.m), axisScale, axisOffset);
//					typename SIMD<_SIMDType>::Float yf = SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::convert(yi.m), axisScale, axisOffset);
//					typename SIMD<_SIMDType>::Float zf = SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::convert(zi.m), axisScale, axisOffset);
//
//					xi.m = SIMD<_SIMDType>::add(xi.m, xSIMD);
//					yi.m = SIMD<_SIMDType>::add(yi.m, ySIMD);
//					zi.m = SIMD<_SIMDType>::add(zi.m, zSIMD);
//
//                    uSIMD<SIMD<_SIMDType>::Float, SIMD<_SIMDType>::vectorSize()> sampledResults;
//					sampledResults.m = Lerp<_SIMDType>(
//						Lerp<_SIMDType>(
//							Lerp<_SIMDType>(c000, c100, xf),
//							Lerp<_SIMDType>(c010, c110, xf), yf),
//						Lerp<_SIMDType>(
//							Lerp<_SIMDType>(c001, c101, xf),
//							Lerp<_SIMDType>(c011, c111, xf), yf), zf);
//
//					for (int i = 0; i < vMax; i++)
//					{
//						if (xi.a[i] >= 0 && xi.a[i] < xSize &&
//							yi.a[i] >= 0 && yi.a[i] < ySize &&
//							zi.a[i] >= 0 && zi.a[i] < zSize)
//						{
//							int index = SET_INDEX(xi.a[i], yi.a[i], zi.a[i]);
//							noiseSet[index] = sampledResults.a[i];
//						}
//					}
//
//					localCount += SIMD<_SIMDType>::vectorSize();
//					localCountSIMD = SIMD<_SIMDType>::add(localCountSIMD, Constant::numi_vectorSize);
//				}
//				zSIMD = SIMD<_SIMDType>::add(zSIMD, sampleSizeSIMD);
//			}
//			ySIMD = SIMD<_SIMDType>::add(ySIMD, sampleSizeSIMD);
//		}
//		xSIMD = SIMD<_SIMDType>::add(xSIMD, sampleSizeSIMD);
//	}
//
//	FreeNoiseSet(noiseSetSample);
//	SIMD<_SIMDType>::zeroAll();
//}
//
//template<SIMDType _SIMDType>
//void NoiseSIMD<_SIMDType>::FillSampledNoiseSet(float* noiseSet, FastNoiseVectorSet* vectorSet, float xOffset, float yOffset, float zOffset)
//{
//    typedef Constants<typename SIMD<_SIMDType>::Float, typename SIMD<_SIMDType>::Int, _SIMDType> Constant;
//
//	assert(noiseSet);
//	assert(vectorSet);
//	assert(vectorSet->size >= 0);
//	SIMD<_SIMDType>::zeroAll();
//
//	int sampleScale = vectorSet->sampleScale;
//
//	if (sampleScale <= 0)
//	{
//		FillNoiseSet(noiseSet, vectorSet, xOffset, yOffset, zOffset);
//		return;
//	}
//
//	int sampleSize = 1 << sampleScale;
//	int sampleMask = sampleSize - 1;
//	float scaleModifier = float(sampleSize);
//
//	int xSize = vectorSet->sampleSizeX;
//	int ySize = vectorSet->sampleSizeY;
//	int zSize = vectorSet->sampleSizeZ;
//
//	int xSizeSample = xSize;
//	int ySizeSample = ySize;
//	int zSizeSample = zSize;
//
//	if (xSizeSample & sampleMask)
//		xSizeSample = (xSizeSample & ~sampleMask) + sampleSize;
//
//	if (ySizeSample & sampleMask)
//		ySizeSample = (ySizeSample & ~sampleMask) + sampleSize;
//
//	if (zSizeSample & sampleMask)
//		zSizeSample = (zSizeSample & ~sampleMask) + sampleSize;
//
//	xSizeSample = (xSizeSample >> sampleScale) + 1;
//	ySizeSample = (ySizeSample >> sampleScale) + 1;
//	zSizeSample = (zSizeSample >> sampleScale) + 1;
//
//	float* noiseSetSample = GetEmptySet(vectorSet->size);
//	FillNoiseSet(noiseSetSample, vectorSet, xOffset - 0.5f, yOffset - 0.5f, zOffset - 0.5f);
//
//	int yzSizeSample = ySizeSample * zSizeSample;
//	int yzSize = ySize * zSize;
//
//	typename SIMD<_SIMDType>::Int axisMask = SIMD<_SIMDType>::set(sampleMask);
//	typename SIMD<_SIMDType>::Float axisScale = SIMD<_SIMDType>::set(1.f / scaleModifier);
//	typename SIMD<_SIMDType>::Float axisOffset = SIMD<_SIMDType>::mulf(axisScale, Constant::numf_0_5);
//
//	typename SIMD<_SIMDType>::Int sampleSizeSIMD =SIMD<_SIMDType>::set(sampleSize);
//	typename SIMD<_SIMDType>::Int xSIMD =SIMD<_SIMDType>::zeroInt();
//
//	int localCountMax = (1 << (sampleScale * 3));
//	int vMax = SIMD<_SIMDType>::vectorSize();
//
//#if SIMD_LEVEL == FN_NEON
//	typename SIMD<_SIMDType>::Int sampleScaleV = SIMD<_SIMDType>::set(-sampleScale);
//	typename SIMD<_SIMDType>::Int sampleScale2V = SIMD<_SIMDType>::mul(sampleScaleV, Constant::numi_2);
//#endif
//
//	for (int x = 0; x < xSizeSample - 1; x++)
//	{
//		typename SIMD<_SIMDType>::Int ySIMD = SIMD<_SIMDType>::zeroInt();
//		for (int y = 0; y < ySizeSample - 1; y++)
//		{
//			typename SIMD<_SIMDType>::Int zSIMD =SIMD<_SIMDType>::zeroInt();
//
//			typename SIMD<_SIMDType>::Float c001 =SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y, 0)]);
//			typename SIMD<_SIMDType>::Float c101 =SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y, 0)]);
//			typename SIMD<_SIMDType>::Float c011 =SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y + 1, 0)]);
//			typename SIMD<_SIMDType>::Float c111 =SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, 0)]);
//			for (int z = 0; z < zSizeSample - 1; z++)
//			{
//				typename SIMD<_SIMDType>::Float c000 = c001;
//				typename SIMD<_SIMDType>::Float c100 = c101;
//				typename SIMD<_SIMDType>::Float c010 = c011;
//				typename SIMD<_SIMDType>::Float c110 = c111;
//				c001 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y, z + 1)]);
//				c101 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y, z + 1)]);
//				c011 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x, y + 1, z + 1)]);
//				c111 = SIMD<_SIMDType>::set(noiseSetSample[SAMPLE_INDEX(x + 1, y + 1, z + 1)]);
//
//				typename SIMD<_SIMDType>::Int localCountSIMD = Constant::numi_incremental;
//
//				int localCount = 0;
//				while (localCount < localCountMax)
//				{
//                    uSIMD<SIMD<_SIMDType>::Int, SIMD<_SIMDType>::vectorSize()> xi, yi, zi;
//
//#if SIMD_LEVEL == FN_NEON
//					xi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftL(localCountSIMD, sampleScale2V), axisMask);
//					yi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftL(localCountSIMD, sampleScaleV), axisMask);
//#else
//					xi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(localCountSIMD, sampleScale * 2), axisMask);
//					yi.m = SIMD<_SIMDType>::_and(SIMD<_SIMDType>::shiftR(localCountSIMD, sampleScale), axisMask);
//#endif
//
//					zi.m = SIMD<_SIMDType>::_and(localCountSIMD, axisMask);
//
//					typename SIMD<_SIMDType>::Float xf = SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::convert(xi.m), axisScale, axisOffset);
//					typename SIMD<_SIMDType>::Float yf = SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::convert(yi.m), axisScale, axisOffset);
//					typename SIMD<_SIMDType>::Float zf = SIMD<_SIMDType>::mulAdd(SIMD<_SIMDType>::convert(zi.m), axisScale, axisOffset);
//
//					xi.m = SIMD<_SIMDType>::add(xi.m, xSIMD);
//					yi.m = SIMD<_SIMDType>::add(yi.m, ySIMD);
//					zi.m = SIMD<_SIMDType>::add(zi.m, zSIMD);
//
//                    uSIMD<SIMD<_SIMDType>::Float, SIMD<_SIMDType>::vectorSize()> sampledResults;
//					sampledResults.m = Lerp<_SIMDType>(
//						Lerp<_SIMDType>(
//							Lerp<_SIMDType>(c000, c100, xf),
//							Lerp<_SIMDType>(c010, c110, xf), yf),
//						Lerp<_SIMDType>(
//							Lerp<_SIMDType>(c001, c101, xf),
//							Lerp<_SIMDType>(c011, c111, xf), yf), zf);
//
//					for (int i = 0; i < vMax; i++)
//					{
//						if (xi.a[i] < xSize &&
//							yi.a[i] < ySize &&
//							zi.a[i] < zSize)
//						{
//							int index = SET_INDEX(xi.a[i], yi.a[i], zi.a[i]);
//							noiseSet[index] = sampledResults.a[i];
//						}
//					}
//
//					localCount += SIMD<_SIMDType>::vectorSize();
//					localCountSIMD = SIMD<_SIMDType>::add(localCountSIMD, Constant::numi_vectorSize);
//				}
//				zSIMD = SIMD<_SIMDType>::add(zSIMD, sampleSizeSIMD);
//			}
//			ySIMD = SIMD<_SIMDType>::add(ySIMD, sampleSizeSIMD);
//		}
//		xSIMD = SIMD<_SIMDType>::add(xSIMD, sampleSizeSIMD);
//	}
//
//	FreeNoiseSet(noiseSetSample);
//	SIMD<_SIMDType>::zeroAll();
//}

}//namespace details
}//namespace FastNoise