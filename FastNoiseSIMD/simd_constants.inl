namespace FastNoise
{
namespace details
{

template<typename Float, typename Int>
struct ConstantsBase
{
    static bool valueSet;

    static Int numi_0xffffffff;
    static Float numf_1;

    static Float numf_incremental;
    static Float numf_0;
    static Float numf_2;
    static Float numf_6;
    static Float numf_10;
    static Float numf_15;
    static Float numf_32;
    static Float numf_999999;

    static Float numf_0_5;
    static Float numf_0_6;
    static Float numf_15_5;
    static Float numf_511_5;

    static Float numf_F3;
    static Float numf_G3;
    static Float numf_G33;
    static Float numf_hash2Float;
    static Float numf_vectorSize;
    static Float numf_cubicBounding;

    static Int numi_incremental;
    static Int numi_1;
    static Int numi_2;
    static Int numi_255;
    static Int numi_60493;
    static Int numi_0x7fffffff;

    static Int numi_xPrime;
    static Int numi_yPrime;
    static Int numi_zPrime;
    static Int numi_bit5Mask;
    static Int numi_bit10Mask;
    static Int numi_vectorSize;
};

template<typename Float, typename Int>
bool ConstantsBase<Float, Int>::valueSet;

template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_0xffffffff;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_1;

template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_incremental;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_0;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_2;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_6;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_10;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_15;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_32;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_999999;

template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_0_5;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_0_6;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_15_5;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_511_5;

template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_F3;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_G3;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_G33;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_hash2Float;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_vectorSize;
template<typename Float, typename Int>
Float ConstantsBase<Float, Int>::numf_cubicBounding;

template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_incremental;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_1;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_2;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_255;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_60493;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_0x7fffffff;

template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_xPrime;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_yPrime;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_zPrime;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_bit5Mask;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_bit10Mask;
template<typename Float, typename Int>
Int ConstantsBase<Float, Int>::numi_vectorSize;

template<typename Float, typename Int, SIMDType _SIMDType>
struct Constants:ConstantsBase<Float, Int>
{
    static Int numi_8;
    static Int numi_12;
    static Int numi_13;
};

template<typename Float, typename Int, SIMDType _SIMDType>
Int Constants<Float, Int, _SIMDType>::numi_8;
template<typename Float, typename Int, SIMDType _SIMDType>
Int Constants<Float, Int, _SIMDType>::numi_12;
template<typename Float, typename Int, SIMDType _SIMDType>
Int Constants<Float, Int, _SIMDType>::numi_13;

}//namespace details
}//namespace FastNoise
