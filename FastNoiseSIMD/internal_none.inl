#include <math.h>

namespace FastNoise
{
namespace details
{

template<SIMDType _SIMDType>
struct SIMD
{
    typedef float Float;
    typedef int Int;
    typedef Int Mask;

    static Int numi_8;
    static Int numi_12;
    static Int numi_13;

    static constexpr size_t const vectorSize() { return 1; }
    static constexpr size_t const alignment() { return 4; }
    static constexpr size_t const level() { return (size_t)SIMDType::None; }

    static Float set(float a) { return a; }
    static Float zeroFloat() { return 0.0f; }
    static Int set(int a) { return a; }
    static Int zeroInt() { return 0; }
    static void zeroAll() {}

    static void store(float *p, Float a) { *(p)=a; }
    static Float load(float *p) { return *p; }

    static Float undefinedFloat() { return zeroFloat(); }
    static Int undefinedInt() { return zeroInt(); }

    static Float convert(Int a) { return static_cast<float>(a); }
    static Float cast(Int a) { return *reinterpret_cast<float*>(&a); }
    static Int convert(Float a)
    {
        return static_cast<int>(roundf(a));
    }
    static Int cast(Float a) { return  *reinterpret_cast<int*>(&a); }

    static Float add(Float a, Float b) { return a+b; }
    static Float sub(Float a, Float b) { return a-b; }
    static Float mulf(Float a, Float b) { return a*b; }
    static Float div(Float a, Float b) { return a/b; }

    static Float min(Float a, Float b) { return fminf(a, b); }
    static Float max(Float  a, Float b) { return fmaxf(a, b); }
    static Float invSqrt(Float x)
    {
        //this does not match the precision of the SIMD functions, nor will 1.0f/sqrt(x)
        float xhalf=0.5f * x;
        int i=*(int*)&x;
        i=0x5f3759df-(i>>1);
        x=*(float*)&i;
        x=x*(1.5f-xhalf*x*x);
        return x;
    }

    static Mask equal(Float a, Float b) { return ((a)==(b))?0xFFFFFFFF:0; }
    static Mask lessThan(Float a, Float b) { return ((a)<(b))?0xFFFFFFFF:0; }
    static Mask greaterThan(Float a, Float b) { return ((a)>(b))?0xFFFFFFFF:0; }
    static Mask lessEqual(Float a, Float b) { return ((a)<=(b))?0xFFFFFFFF:0; }
    static Mask greaterEqual(Float a, Float b) { return ((a)>=(b))?0xFFFFFFFF:0; }

    static Float _and(Float a, Float b) { return cast(cast(a) & cast(b)); }
    static Float andNot(Float a, Float b) { return cast(~cast(a) & cast(b)); }
    static Float _xor(Float a, Float b) { return cast(cast(a)^cast(b)); }

    static Float floor(Float a) { return floorf(a); }
    static Float abs(Float a) { return fabsf(a); }
    static Float blend(Float a, Float b, Mask mask) { return mask?(b):(a); }

    static Int add(Int a, Int b) { return a+b; }
    static Int sub(Int a, Int b) { return a-b; }
    static Int mul(Int a, Int b) { return a*b; }

    static Float mulAdd(Float a, Float b, Float c) { return add(mulf(a, b), c); }
    static Float nmulAdd(Float a, Float b, Float c) { return sub(c, mulf(a, b)); }
    static Float mulSub(Float a, Float b, Float c) { return sub(mulf(a, b), c); }

    static Int _and(Int a, Int b) { return a&b; }
    static Int andNot(Int a, Int b) { return (~a)&b; }
    static Int _or(Int a, Int b) { return a|b; }
    static Int _xor(Int a, Int b) { return a^b; }
    static Int _not(Int a) { return (~a); }

    static Int shiftR(Int a, Int b) { return a>>b; }
    static Int shiftL(Int a, Int b) { return a<<b; }

    static Mask equal(Int a, Int b) { return ((a)==(b))?0xFFFFFFFF:0; }
    static Mask greaterThan(Int a, Int b) { return ((a)>(b))?0xFFFFFFFF:0; }
    static Mask lessThan(Int a, Int b) { return ((a)<(b))?0xFFFFFFFF:0; }

    static Mask maskAnd(Mask a, Mask b) { return _and(a,b); }
    static Mask maskAndNot(Mask a, Mask b) { return andNot(a, b); }
    static Mask maskOr(Mask a, Mask b) { return _or(a,b); }
    static Mask maskNot(Mask a) { return _not(a); }

    static Float mask(Mask m, Float a) { return _and(cast(m), a); }
    static Float maskAdd(Mask m, Float a, Float b) { return add(a, _and(cast(m), b)); }
    static Float maskSub(Mask m, Float a, Float b) { return sub(a, _and(cast(m), b)); }

    static Int maskAdd(Mask m, Int a, Int b) { return add(a, _and(m, b)); }
    static Int maskSub(Mask m, Int a, Int b) { return sub(a, _and(m, b)); }
};

}//namespace details
}//namespace FastNoise
