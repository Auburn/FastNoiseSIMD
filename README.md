# FastNoise SIMD
FastNoise SIMD is the SIMD implementation of my noise library [FastNoise](https://github.com/Auburns/FastNoise). It aims to provide faster performance through the use of intrinsic(SIMD) CPU functions. Vectorisation of the code allows noise functions to process data in sets of 4/8/16 increasing performance by 700% in some cases (Simplex).

After releasing FastNoise I got in contact with the author of [FastNoise SIMD](https://github.com/jackmott/FastNoise-SIMD) (naming is coincidence) and was inspired to work with SIMD functions myself. Through his code and discussions with him I created my implementation with even more optimisation thanks to the removal of lookup tables. 

Runtime detection of highest supported instruction set ensures the fastest possible performance with only 1 compile needed. If no support is found it will fallback to standard types (float/int).

## Features

- Value Noise 3D
- Perlin Noise 3D
- Simplex Noise 3D
- Multiple fractal options for all of the above
- White Noise 3D
- Cellular Noise 3D
- Cubic Noise 3D
- Large variety of cellular noise settings
- Perturb input coordinates in 3D space
- Integrated noise sampling

## Supported Instruction Sets
- ARM NEON
- AVX512
- AVX2 - FMA3
- SSE4.1
- SSE2

## Tested Compilers
- MSVC v120/v140
- Intel 16.0
- GCC 4.7 Linux
- Clang MacOSX

## Wiki
[Getting Started](https://github.com/Auburns/FastNoiseSIMD/wiki)

More documentation coming soon.

# FastNoise SIMD Preview

I have written a compact testing application for all the features included in FastNoiseSIMD with a visual representation. I use this for development purposes and testing noise settings used in terrain generation. The fastest supported instruction set is also reported.

Download links can be found in the [Releases Section](https://github.com/Auburns/FastNoiseSIMD/releases).

![Simplex Fractal](http://i.imgur.com/45JkT5j.png)

# Performance Comparisons
Using default noise settings on FastNoise SIMD and matching those settings across the other libraries where possible.

Timings below are x1000 ns to generate 32x32x32 points of noise.

- CPU: Intel Xeon Skylake @ 2.0Ghz
- Compiler: Intel 17.0 x64

| Noise Type  | AVX512 | AVX2 | SSE4.1 | SSE2 | FastNoise | LibNoise |
|-------------|--------|------|--------|------|-----------|----------|
| White Noise | 9      | 13   | 23     | 69   | 211       |          |
| Value       | 110    | 147  | 327    | 760  | 663       |          |
| Perlin      | 167    | 334  | 624    | 1021 | 952       | 1374     |
| Simplex     | 173    | 338  | 592    | 732  | 1246      |          |
| Cellular    | 1055   | 1406 | 3299   | 4024 | 3039      | 59688    |
| Cubic       | 1001   | 1349 | 2586   | 5672 | 2968      |          |

Comparision of fractals and sampling performance [here](https://github.com/Auburns/FastNoiseSIMD/wiki/In-depth-SIMD-level).

# Examples
### Cellular Noise
![Cellular Noise](http://i.imgur.com/RshUkoe.png)

![Cellular Noise](http://i.imgur.com/PjPYBXu.png)

![Cellular Noise](http://i.imgur.com/hyKjIuH.png)

### Fractal Noise
![Simplex Fractal Billow](http://i.imgur.com/gURJtpc.png)

![Perlin Fractal Billow](http://i.imgur.com/IcjbpYz.png)

### Value Noise
![Value Noise](http://i.imgur.com/Ss22zRs.png)

### White Noise
![White Noise](http://i.imgur.com/wcTlyek.png)

### Perturb
![Perturbed Cellular Noise](http://i.imgur.com/xBKGo1E.png)

