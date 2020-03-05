#include <catch2/catch.hpp>

#include "FastNoiseSIMD/FastNoiseSIMD.h"

TEST_CASE("simplex", "[FastNoiseSIMD]")
{
    FastNoiseSIMD* noise = FastNoiseSIMD::NewFastNoiseSIMD();

    int x = 16;
    int y = 16;
    int z = 16;
    int x_size = 8;
    int y_size = 8;
    int z_size = 8;
    float* simplex_set = noise->GetSimplexSet(x, y, z, x_size, y_size, z_size);

    noise->FreeNoiseSet(simplex_set);
    delete noise;
}
