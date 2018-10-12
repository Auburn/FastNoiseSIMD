cmake_minimum_required(VERSION 3.1)

project(FastNoise)

set(FN_SIMD_SSE2 OFF)
set(FN_SIMD_SSE41 OFF)
set(FN_SIMD_AVX2 OFF)
set(FN_SIMD_AVX512 OFF)
set(FN_SIMD_NEON OFF)

set(FN_NO_FILESYSTEM 0)
set(FN_STD_FILESYSTEM 1)
set(FN_EXPERIMENTAL_FILESYSTEM 2)
set(FN_USE_FILESYSTEM ${FN_NO_FILESYSTEM})

set(FN_ARM64 OFF)

if(ANDROID_NDK_VERSION)
    string(SUBSTRING ${CMAKE_ANDROID_ARCH_ABI} 0 5 FM_ANDROID_ARCH)
    string(TOLOWER ${FM_ANDROID_ARCH} FM_ANDROID_ARCH)
    string(COMPARE EQUAL ${FM_ANDROID_ARCH} "arm64" FN_ARM64)
elseif(IOS_SDK_VERSION)
    string(SUBSTRING ${IPHONEOS_ARCHS} 0 3 FM_IOS_ARCH)
    string(TOLOWER ${FM_IOS_ARCH} FM_IOS_ARCH)
    string(COMPARE EQUAL ${FM_IOS_ARCH} "arm" FN_ARM64)
endif()

if(FN_ARM64)
    set(FN_SIMD_NEON ON)
else()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        add_definitions(-DNOMINMAX)
    
        set(FN_SIMD_SSE2 ON)
        set(FN_SIMD_SSE41 ON)

        if(MSVC_VERSION EQUAL 1900)
            set(FN_SIMD_AVX2 ON)
            set(FN_USE_FILESYSTEM ${FN_EXPERIMENTAL_FILESYSTEM})
        elseif(MSVC_VERSION GREATER 1900)
    #        set(FN_SIMD_AVX512 ON) something wrong finding floor function
            set(FN_USE_FILESYSTEM ${FN_STD_FILESYSTEM})
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        set(FN_SIMD_SSE2 ON)
        set(FN_SIMD_SSE41 ON)
    
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.9)
            set(FN_SIMD_AVX2 ON)
            set(FN_SIMD_AVX512 ON)
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(FN_SIMD_SSE2 ON)
        set(FN_SIMD_SSE41 ON)
        set(FN_USE_FILESYSTEM ${FN_EXPERIMENTAL_FILESYSTEM})

        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.6)
            set(FN_SIMD_AVX2 ON)
    
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.9)
                set(FN_SIMD_AVX512 ON)
            endif()

            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
                set(FN_USE_FILESYSTEM ${FN_STD_FILESYSTEM})
            endif()
        endif()
    endif()
endif()

set(fastNoise_sources
    FastNoiseSIMD/FastNoiseSIMD.cpp
    FastNoiseSIMD/FastNoiseSIMD_none.cpp
)

set(fastNoise_headers
    FastNoiseSIMD/FastNoiseSIMD.h
    FastNoiseSIMD/FastNoiseSIMD_internal.h
    FastNoiseSIMD/FastNoiseSIMD_internal.inl
    FastNoiseSIMD/simd_constants.inl
    FastNoiseSIMD/internal_none.inl
    FastNoiseSIMD/simd_init.inl
)

add_library(fastNoise SHARED
    ${fastNoise_sources}
    ${fastNoise_headers}
)

target_compile_definitions(fastNoise PRIVATE "-DFN_USE_FILESYSTEM=${FN_USE_FILESYSTEM}")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    target_compile_options(fastNoise PUBLIC "-ldl")
    if(FN_USE_FILESYSTEM GREATER 0)
        target_compile_options(fastNoise PUBLIC "-lstdc++fs")
    endif()
endif()
target_include_directories(fastNoise PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(FN_SIMD_SSE2)
    set(fastNoise_sse2_sources
        FastNoiseSIMD/FastNoiseSIMD_sse2.cpp
    )

    set(fastNoise_sse2_headers
        FastNoiseSIMD/internal_sse2.inl
    )

    add_library(fastNoise_sse2 SHARED
        ${fastNoise_sse2_sources}
        ${fastNoise_sse2_headers}
    )

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        target_compile_options(fastNoise_sse2 PRIVATE "-msse2")
    endif()
    target_link_libraries(fastNoise_sse2 fastNoise)
endif()

if(FN_SIMD_SSE41)
    set(fastNoise_sse41_sources
        FastNoiseSIMD/FastNoiseSIMD_sse41.cpp
    )

    set(fastNoise_sse41_headers
        FastNoiseSIMD/internal_sse41.inl
    )

    add_library(fastNoise_sse41 SHARED
        ${fastNoise_sse41_sources}
        ${fastNoise_sse41_headers}
    )

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        target_compile_options(fastNoise_sse41 PRIVATE "-msse4.1")
    endif()
    target_link_libraries(fastNoise_sse41 fastNoise)
endif()

if(FN_SIMD_AVX2)
    set(fastNoise_avx2_sources
        FastNoiseSIMD/FastNoiseSIMD_avx2.cpp
    )

    set(fastNoise_avx2_headers
        FastNoiseSIMD/internal_avx2.inl
    )

    add_library(fastNoise_avx2 SHARED
        ${fastNoise_avx2_sources}
        ${fastNoise_avx2_headers}
    )

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        target_compile_options(fastNoise_avx2 PRIVATE "/arch:AVX2")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        target_compile_options(fastNoise_avx2 PRIVATE "-mavx2" "-mfma")
    endif()
    target_link_libraries(fastNoise_avx2 fastNoise)
endif()

if(FN_SIMD_AVX512)
    set(fastNoise_avx512_sources
        FastNoiseSIMD/FastNoiseSIMD_avx512.cpp
    )

    set(fastNoise_avx512_headers
        FastNoiseSIMD/internal_avx512.inl
    )

    add_library(fastNoise_avx512 SHARED
        ${fastNoise_avx512_sources}
        ${fastNoise_avx512_headers}
    )

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        target_compile_options(fastNoise_avx512 PRIVATE "/arch:AVX512")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        target_compile_options(fastNoise_avx512 PRIVATE "-mavx512f" "-mfma")
    endif()
    target_link_libraries(fastNoise_avx512 fastNoise)
endif()

add_executable(testApp testApp/main.cpp)

set(fastNoise_libs
    fastNoise
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(fastNoise_libs ${fastNoise_libs}
        Mincore.lib
    )
endif()
target_link_libraries(testApp ${fastNoise_libs})
target_compile_definitions(testApp PRIVATE "-DFN_USE_FILESYSTEM=${FN_USE_FILESYSTEM}")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if(FN_USE_FILESYSTEM GREATER 0)
        target_compile_options(testApp PUBLIC "-lstdc++fs")
    endif()
endif()


if(FN_SIMD_SSE2)
    add_dependencies(testApp fastNoise_sse2)
endif()
if(FN_SIMD_SSE41)
    add_dependencies(testApp fastNoise_sse41)
endif()
if(FN_SIMD_AVX2)
    add_dependencies(testApp fastNoise_avx2)
endif()
if(FN_SIMD_AVX512)
    add_dependencies(testApp fastNoise_avx512)
endif()
