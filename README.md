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
- Cellular Noise 3D
- Large variety of cellular noise settings

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

![Simplex Fractal](http://i.imgur.com/45JkT5j.png)

#Performance Comparisons
Using default noise settings on FastNoise SIMD and matching those settings across the other libraries where possible.

Timings below are average milliseconds for 1 million iterations on a single thread.

- CPU: i7 4790k @ 4.0Ghz
- Compiler: MSVC v140 x64

| Noise Type       | FastNoise SIMD - AVX2 | FastNoise | LibNoise | ANL      |
|------------------|-----------------------|-----------|----------|----------|
| Value            | 4.12                  | 13.85     | 24.16    | 94.73    |
| Value Fractal    | 12.22                 | 46.99     |          | 289.91   |
| Gradient         | 7.68                  | 21.69     | 32.68    | 109.26   |
| Gradient Fractal | 21.54                 | 84.22     | 122.15   | 325.04   |
| Simplex          | 7.03                  | 27.56     |          | 43.68    |
| Simplex Fractal  | 19.91                 | 85.47     |          | 154.41   |
| White Noise      | 0.47                  | 2.81      |          |          |
| Cellular         | 31.42                 | 122.21    | 1,122.60 | 2,473.06 |

Comparision of different SIMD levels can be seen [here](https://github.com/Auburns/FastNoiseSIMD/wiki/In-depth-SIMD-level).

#Examples
###Cellular Noise
![Cellular Noise](http://i.imgur.com/RshUkoe.png)

![Cellular Noise](http://i.imgur.com/PjPYBXu.png)

![Cellular Noise](http://i.imgur.com/hyKjIuH.png)

###Fracal Noise
![Simplex Fractal Billow](http://i.imgur.com/gURJtpc.png)

![Gradient Fractal Billow](http://i.imgur.com/IcjbpYz.png)

###Value Noise
![Value Noise](http://i.imgur.com/Ss22zRs.png)

###White Noise
![White Noise](http://i.imgur.com/wcTlyek.png)
