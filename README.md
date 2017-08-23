# FastNoise SIMD
FastNoise SIMD is the SIMD implementation of my noise library [FastNoise](https://github.com/Auburns/FastNoise). It aims to provide faster performance through the use of intrinsic(SIMD) CPU functions. Vectorisation of the code allows noise functions to process data in sets of 4/8/16 increasing performance by 700% in some cases (Simplex).

After releasing FastNoise I got in contact with the author of [FastNoise SIMD](https://github.com/jackmott/FastNoise-SIMD) (naming is coincidence) and was inspired to work with SIMD functions myself. Through his code and discussions with him I created my implementation with even more optimisation thanks to the removal of lookup tables. 

Runtime detection of highest supported instruction set ensures the fastest possible performance with only 1 compile needed. If no support is found it will fallback to standard types (float/int).

## Features

- Value Noise 3D
- Perlin Noise 3D
- Simplex Noise 3D
- Cubic Noise 3D
- Multiple fractal options for all of the above
- White Noise 3D
- Cellular Noise 3D
- Perturb input coordinates in 3D space
- Integrated up-sampling
- Easy to use 3D cave noise

Credit to [CubicNoise](https://github.com/jobtalle/CubicNoise) for the cubic noise algorithm

## Supported Instruction Sets
- ARM NEON
- AVX-512F
- AVX2 - FMA3
- SSE4.1
- SSE2

## Tested Compilers
- MSVC v120/v140
- Intel 16.0
- GCC 4.7 Linux
- Clang MacOSX

## Wiki
[Docs](https://github.com/Auburns/FastNoiseSIMD/wiki)

## Related repositories

- [FastNoise](https://github.com/Auburns/FastNoise)
- [PyFastNoiseSIMD](https://github.com/robbmcleod/PyFastNoiseSIMD)

# FastNoise SIMD Preview

I have written a compact testing application for all the features included in FastNoiseSIMD with a visual representation. I use this for development purposes and testing noise settings used in terrain generation. The fastest supported instruction set is also reported.

Download links can be found in the [Releases Section](https://github.com/Auburns/FastNoiseSIMD/releases).

![Simplex Fractal](http://i.imgur.com/45JkT5j.png)

# Performance Comparisons
Using default noise settings on FastNoise SIMD and matching those settings across the other libraries where possible.

Timings below are x1000 ns to generate 32x32x32 points of noise on a single thread.

- CPU: Intel Xeon Skylake @ 2.0Ghz
- Compiler: Intel 17.0 x64

| Noise Type  | AVX512 | AVX2 | SSE4.1 | SSE2 | FastNoise | LibNoise |
|-------------|--------|------|--------|------|-----------|----------|
| White Noise | 7      | 9    | 16     | 29   | 141       |          |
| Value       | 92     | 152  | 324    | 436  | 642       |          |
| Perlin      | 147    | 324  | 592    | 795  | 1002      | 1368     |
| Simplex     | 129    | 294  | 548    | 604  | 1194      |          |
| Cellular    | 851    | 1283 | 2679   | 2959 | 2979      | 58125    |
| Cubic       | 615    | 952  | 1970   | 3516 | 2979      |          |

Comparision of fractals and sampling performance [here](https://github.com/Auburns/FastNoiseSIMD/wiki/In-depth-SIMD-level).

# Examples
### Cellular Noise
![Cellular Noise](http://i.imgur.com/RshUkoe.png)

![Cellular Noise](http://i.imgur.com/PjPYBXu.png)

![Cellular Noise](http://i.imgur.com/hyKjIuH.png)

[Cave noise example](https://www.youtube.com/watch?v=Df4Hidvq11M)

### Fractal Noise
![Simplex Fractal Billow](http://i.imgur.com/gURJtpc.png)

![Perlin Fractal Billow](http://i.imgur.com/IcjbpYz.png)

### Value Noise
![Value Noise](http://i.imgur.com/Ss22zRs.png)

### White Noise
![White Noise](http://i.imgur.com/wcTlyek.png)

### Perturb
![Perturbed Cellular Noise](http://i.imgur.com/xBKGo1E.png)

