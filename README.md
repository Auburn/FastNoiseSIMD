# FastNoise SIMD
FastNoise SIMD is the SIMD implementation of my noise library [FastNoise](https://github.com/Auburns/FastNoise). It aims to provide faster performance through the use of intrinsic(SIMD) CPU functions. Vectorisation of the code allows noise functions to process data in sets of 4 or 8 increasing performance by 300-400%.

After releasing FastNoise I got in contact with the author of [FastNoise SIMD](https://github.com/jackmott/FastNoise-SIMD) (naming is coincidence) and was inspired to work with SIMD functions myself. Through his code and discussions with him I created my implementation with even more optimisation thanks to the removal of lookup tables. 

Runtime detection of highest supported instruction set ensures the fastest possible performance with only 1 compile needed. If no support is found it will fallback to standard types (float/int).

##Features

- Value Noise 3D
- Gradient (Perlin) Noise 3D
- Simplex Noise 3D
- Multiple fractal options for all of the above
- White Noise 3D

##Supported Instruction Sets
- AVX2 - FMA3
- SSE4.1
- SSE2

##Wiki
[Getting Started](https://github.com/Auburns/FastNoiseSIMD/wiki)

More documentation coming soon.

#FastNoise SIMD Preview

I have written a compact testing application for all the features included in FastNoiseSIMD with a visual representation. I use this for development purposes and testing noise settings used in terrain generation. The fastest supported instruction set is also reported.

Download links can be found in the [Releases Section](https://github.com/Auburns/FastNoiseSIMD/releases).

![FastNoise SIMD Preview](http://i.imgur.com/p4kHh7P.png)

#Performance Comparisons
Using default noise settings on FastNoise SIMD and matching those settings across the other libraries where possible.

Timing are averages of time taken for 1 million iterations on a single thread.

- CPU: i7 4790k @ 4.0Ghz
- Compiler: MSVC v120

Noise Type |	FastNoise SIMD (AVX2) |	FastNoise |	LibNoise |	ANL
| ---               | ---     | ---     | ---     | ---     |
| Value             | 4.48    | 15.68   | 25.08   | 98.21   |
| Value Fractal     | 13.49   | 52.79   |         | 310.84  |
| Gradient          | 8.23    | 24.47   | 29.68   | 103.06  |
| Gradient Fractal  | 26.80   | 80.65   | 109.67  | 325.56  |
| Simplex           | 7.76    | 30.84   |         |  43.48  |
| Simplex Fractal   | 23.43   | 105.07  |         | 162.27  |
| White Noise       | 0.48    | 2.92    |         |         |
| Cellular          |         | 123.12  |         |         |
| Cellular HQ       |         | 481.04  | 1,130.63 | 2,393.92 |

Measurements are in ms
