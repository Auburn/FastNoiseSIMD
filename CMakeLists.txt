cmake_minimum_required(VERSION 3.10)

# detect if FastNoiseSIMD is being bundled,
# disable testsuite in that case
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
endif()

project(FastNoiseSIMD
    VERSION 0.7.0
    DESCRIPTION "C++ SIMD Noise Library"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(BUILD_TESTING)
    # Catch2 needed for testing
    find_package(Catch2 CONFIG REQUIRED)
endif()

# Do stuff depending on the compiler
option(FN_SET_c0rp3n_CXX_FLAGS
    "Set CMAKE_CXX_FLAGS[_DEBUG][_RELEASE] defaults from the c0rp3n/fastnoise-simd fork."
    OFF)
if(FN_SET_c0rp3n_CXX_FLAGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-W -Wall -Wextra -Wpedantic -Wunused-value -Wold-style-cast")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "/W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/O0 /ZI")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2")
endif()
endif()

set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT FastNoiseSIMD)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    option(FN_COMPILE_NEON
        "Only on arm or aarch64."
        ON)
endif()

option(FN_COMPILE_SSE2 "" ON)

option(FN_COMPILE_SSE41 "" ON)

option(
    FN_COMPILE_AVX2 
    "This does not break support for pre AVX CPUs, AVX code is only run if \
    support is detected." 
    OFF
)

option(FN_COMPILE_AVX512 "Only the latest compilers will support this." OFF)

option(
    FN_USE_FMA 
    "Using FMA instructions with AVX(51)2/NEON provides a small performance \
    increase but can cause minute variations in noise output compared to other \
    SIMD levels due to higher calculation precision."
    ON
)

option(
    FN_ALIGNED_SETS
    "Using aligned sets of memory for float arrays allows faster storing of \
    SIMD data."
    ON
)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/FastNoiseSIMD_config.h.in
    ${CMAKE_CURRENT_LIST_DIR}/include/FastNoiseSIMD/FastNoiseSIMD_config.h
    @ONLY
)

add_library(FastNoiseSIMD STATIC
    src/FastNoiseSIMD.cpp
    src/FastNoiseSIMD_avx2.cpp
    src/FastNoiseSIMD_avx512.cpp
    src/FastNoiseSIMD_internal.cpp
    src/FastNoiseSIMD_neon.cpp
    src/FastNoiseSIMD_sse2.cpp
    src/FastNoiseSIMD_sse41.cpp
)

set_target_properties(FastNoiseSIMD
    PROPERTIES 
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CMAKE_DEBUG_POSTFIX "d"
)

target_include_directories(FastNoiseSIMD PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(FN_CXX_FLAGS)
if(${FN_COMPILE_AVX512})
    if(${MSVC})
        list(APPEND FN_CXX_FLAGS "/arch:AVX512")
    else()
        list(APPEND FN_CXX_FLAGS "-march=skylake-avx512")
    endif()
elseif(${FN_COMPILE_AVX2})
    if(${MSVC})
        list(APPEND FN_CXX_FLAGS "/arch:AVX2")
    else()
        list(APPEND FN_CXX_FLAGS "-march=core-avx2")
    endif()
endif()

target_compile_options(FastNoiseSIMD PRIVATE "${FN_CXX_FLAGS}")

if(BUILD_TESTING)
    include(test/tests.cmake)
endif()

# Only perform the installation steps when not being used as
# a subproject via `add_subdirectory`, or the destinations will break
if(NOT_SUBPROJECT)
    set(FN_CMAKE_CONFIG_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FastNoiseSIMD")

    configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/FastNoiseSIMDConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/FastNoiseSIMDConfig.cmake
        INSTALL_DESTINATION ${FN_CMAKE_CONFIG_DESTINATION}
    )

    # create and install an export set for FastNoiseSIMD target as FastNoiseSIMD
    install(
        TARGETS FastNoiseSIMD EXPORT FastNoiseSIMDTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )


    install(EXPORT FastNoiseSIMDTargets DESTINATION ${FN_CMAKE_CONFIG_DESTINATION})

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/FastNoiseSIMDConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
    )

    install(TARGETS FastNoiseSIMD LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    install(DIRECTORY "include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/FastNoiseSIMDConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/FastNoiseSIMDConfigVersion.cmake"
        DESTINATION ${FN_CMAKE_CONFIG_DESTINATION}
    )
endif()
