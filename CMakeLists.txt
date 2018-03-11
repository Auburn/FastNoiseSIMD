# This is currently very rudimentary and likely needs various improvements.

cmake_minimum_required(VERSION 3.7) # Requiring 3.7 might be incorrect.
project(FastNoiseSIMD)

set(CMAKE_CXX_STANDARD 11) # Requiring C++11 might be incorrect.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(directory ${PROJECT_NAME})
set(library_name ${PROJECT_NAME})
set(file_prefix ${PROJECT_NAME})

set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${library_name})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	option(FN_COMPILE_NEON
		"Only on arm or aarch64."
		ON)
endif()

option(FN_COMPILE_SSE2
	""
	ON)
option(FN_COMPILE_SSE41
	""
	ON)
option(FN_COMPILE_AVX2
	"This does not break support for pre AVX CPUs, AVX code is only run if support is detected."
	OFF)
option(FN_COMPILE_AVX512
	"Only the latest compilers will support this."
	OFF)
option(FN_USE_FMA
	"Using FMA instructions with AVX(51)2/NEON provides a small performance increase but can cause \
minute variations in noise output compared to other SIMD levels due to higher calculation precision."
	ON)
option(FN_ALIGNED_SETS
	"Using aligned sets of memory for float arrays allows faster storing of SIMD data."
	ON)

configure_file(${directory}/${file_prefix}.cmake.h include/${file_prefix}.h)

set(flags)
if(${MSVC})
	if(${FN_COMPILE_AVX2} OR ${FN_COMPILE_AVX512})
		list(APPEND flags "/arch:AVX2")
	endif()
elseif()
	if(${FN_COMPILE_AVX2} OR ${FN_COMPILE_AVX512})
		list(APPEND flags "-march=core-avx2")
	endif()
endif()

add_library(${library_name} STATIC
	${directory}/${file_prefix}.cpp
	${directory}/${file_prefix}_avx2.cpp
	${directory}/${file_prefix}_avx512.cpp
	${directory}/${file_prefix}_internal.cpp
	${directory}/${file_prefix}_neon.cpp
	${directory}/${file_prefix}_sse2.cpp
	${directory}/${file_prefix}_sse41.cpp
	)

target_include_directories(${library_name} PRIVATE
	${CMAKE_CURRENT_BINARY_DIR}/include
	${directory}/
	)

set(public_headers
	${CMAKE_CURRENT_BINARY_DIR}/include/${file_prefix}.h
	# ${directory}/${file_prefix}_internal.h
	)
set_target_properties(${library_name} PROPERTIES
	PUBLIC_HEADER "${public_headers}"
	)

target_compile_options(${library_name} PRIVATE
	"${flags}"
	)

install(TARGETS ${library_name}
	CONFIGURATIONS Debug
	ARCHIVE DESTINATION lib/Debug
	PUBLIC_HEADER DESTINATION include
	)
install(TARGETS ${library_name}
	CONFIGURATIONS Release
	ARCHIVE DESTINATION lib/Release
	PUBLIC_HEADER DESTINATION include
	)
