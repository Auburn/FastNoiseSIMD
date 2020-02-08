#ifndef FASTNOISE_SIMD_CONFIG_H
#define FASTNOISE_SIMD_CONFIG_H

#pragma once

#if defined(__arm__) || defined(__aarch64__)
#define FN_ARM
//#define FN_IOS
/* #undef FN_COMPILE_NEON */
#else

// Comment out lines to not compile for certain instruction sets
#define FN_COMPILE_SSE2
#define FN_COMPILE_SSE41

// To compile AVX2 set C++ code generation to use /arch:AVX(2) on FastNoiseSIMD_avx2.cpp
// Note: This does not break support for pre AVX CPUs, AVX code is only run if support is detected
#define FN_COMPILE_AVX2

// Only the latest compilers will support this
#define FN_COMPILE_AVX512

// Using FMA instructions with AVX(51)2/NEON provides a small performance increase but can cause 
// minute variations in noise output compared to other SIMD levels due to higher calculation precision
// Intel compiler will always generate FMA instructions, use /Qfma- or -no-fma to disable
#define FN_USE_FMA
#endif

// Using aligned sets of memory for float arrays allows faster storing of SIMD data
// Comment out to allow unaligned float arrays to be used as sets
#define FN_ALIGNED_SETS

#endif
