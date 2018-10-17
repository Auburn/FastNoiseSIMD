namespace FastNoise
{
namespace details
{

template<>
struct SIMD<SIMDType::Neon>
{
    typedef float32x4_t Float;
    typedef int32x4_t Int;
    typedef Int Mask;
    static constexpr size_t const vectorSize() { return 4; }
    static constexpr size_t const alignment() { return 16; }
    static constexpr int const level() { return FN_NEON; }

    static Float set(float a) { return vdupq_n_f32(a); }
    static Float zero() { return vdupq_n_f32(0); }
    static Int set(int a) { return vdupq_n_s32(a); }
    static Int zero() { return vdupq_n_s32(0); }
    static void zeroAll() {}

    static void store(float *p, Float a) { vst1q_f32(p, a); }
    static void load(float *p) { return vld1q_f32(p); }

    static Float undefinedFloat() { set(0); }
    static Int undefinedInt() { set(0); }

    static Float convert(Int a) { return vcvtq_f32_s32(a); }
    static Float cast(Int a) { return vreinterpretq_f32_s32(a); }
    static Int convert(Float a) { return vcvtq_s32_f32(a); }
    static Int cast(Float a) { vreinterpretq_s32_f32(a); }

    static Float add(Float a, Float b) { vaddq_f32(a, b); }
    static Float sub(Float a, Float b) { vsubq_f32(a, b); }
    static Float mulf(Float a, Float b) { vmulq_f32(a, b); }
    static Float VECTORCALL v_div(Float a, Float b)
    {
        Float reciprocal=vrecpeq_f32(b);
        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
        reciprocal=vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);

        // and finally, compute a/b = a*(1/b)
        return vmulq_f32(a, reciprocal);
    }
    static Float div(Float a, Float b) { v_div(a, b); }

    static Float mulAdd(Float a, Float b, Float c) { return vmlaq_f32(b, c, a); }
    static Float nmulAdd(Float a, Float b, Float c) { return vmlaq_f32(b, c, a); }
    static Float mulSub(Float a, Float b, Float c) { return sub(mulf(a, b), c); }

    static Float min(Float a, Float b) { vminq_f32(a, b); }
    static Float max(Float  a, Float b) { vmaxq_f32(a, b); }
    static Float invSqrt(Float a) { vrsqrteq_f32(a); }

    static Int lessThan(Float a, Float b) { vreinterpretq_s32_u32(vcltq_f32(a, b)); }
    static Int greaterThan(Float a, Float b) { vreinterpretq_s32_u32(vcgtq_f32(a, b)); }
    static Int lessEqual(Float a, Float b) { vreinterpretq_s32_u32(vcleq_f32(a, b)); }
    static Int greaterEqual(Float a, Float b) { vreinterpretq_s32_u32(vcgeq_f32(a, b)); }

    static Float and(Float a, Float b) { Float cast(vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b))); }
    static Float andNot(Float a, Float b) { Float cast(vandq_s32(vmvnq_s32(vreinterpretq_s32_f32(a)), vreinterpretq_s32_f32(b))); }
    static Float xor(Float a, Float b) { cast(veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b))); }

#ifndef __aarch64__
    static Float VECTORCALL v_floor(Float a)
    {
        Float fval=convert(convert(a));

        return vsubq_f32(fval,
            cast(vandq_s32(lessThan(a, fval),
                cast(SIMDf_NUM(1)))));
    }
    static Float floor(Float a) { v_floor(a); }
#else

    static Float floor(Float a) { vrndmq_f32(a); }
#endif

    static Float abs(Float a) { vabsq_f32(a); }
    static Float blend(Float a, Float b, Mask mask) { vbslq_f32(vreinterpretq_u32_s32(mask), b, a); }

    static Int add(Int a, Int b) { vaddq_s32(a, b); }
    static Int sub(Int a, Int b) { vsubq_s32(a, b); }
    static Int mul(Int a, Int b) { vmulq_s32(a, b); }

    static Int and(Int a, Int b) { vandq_s32(a, b); }
    static Int andNot(Int a, Int b) { vandq_s32(vmvnq_s32(a), b); }
    static Int or(Int a, Int b) { vorrq_s32(a, b); }
    static Int xor(Int a, Int b) { veorq_s32(a, b); }
    static Int not(Int a) { vmvnq_s32(a); }

    static Int shiftR(Int a, int b) { vshrq_n_s32(a, b); }
    static Int shiftL(Int a, int b) { vshlq_n_s32(a, b); }
    static Int vshiftL(Int a, Int b) { vshlq_s32(a, b); }

    static Int equal(Int a, Int b) { vreinterpretq_s32_u32(vceqq_s32(a, b)); }
    static Int greaterThan(Int a, Int b) { vreinterpretq_s32_u32(vcgtq_s32(a, b)); }
    static Int lessThan(Int a, Int b) { vreinterpretq_s32_u32(vcltq_s32(a, b)); }

    static Mask maskAnd(Mask a, Mask b) { return and(a,b); }
    static Mask maskAndNot(Mask a, Mask b) { return andNot(a, b); }
    static Mask maskOr(Mask a, Mask b) { return or(a,b); }
    static Mask maskNot(Mask a) { return not(a); }

    static Float mask(Mask m, Float a) { return and(cast(m), a); }
    static Float maskAdd(Mask m, Float a, Float b) { return add(a, and(cast(m), b)); }
    static Float maskSub(Mask m, Float a, Float b) { return sub(a, and(cast(m), b)); }

    static Int maskAdd(Mask m, Int a, Int b) { return add(a, and(m, b)); }
    static Int maskSub(Mask m, Int a, Int b) { return sub(a, and(m, b)); }
};

}//namespace details
}//namespace FastNoise
