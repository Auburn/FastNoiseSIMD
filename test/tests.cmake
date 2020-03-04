add_executable(FastNoiseSIMD_tests
    test/simplex_noise.cpp
    test/main.cpp
)

target_link_libraries(FastNoiseSIMD_tests 
    FastNoiseSIMD 
    Catch2::Catch2
)

include(CTest)
include(Catch)
catch_discover_tests(FastNoiseSIMD_tests)
