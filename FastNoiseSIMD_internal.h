#ifdef SIMD_LEVEL_H

class FASTNOISE_SIMD_CLASS(SIMD_LEVEL_H) : public FastNoiseSIMD
{
public:
	FASTNOISE_SIMD_CLASS(SIMD_LEVEL_H)(int seed = 1337);

	float* GetGradientSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance = 1.0f) override;
	float* GetGradientFractalSet(int xStart, int yStart, int zStart, int xSize, int ySize, int zSize, float stepDistance = 1.0f) override;
};

#undef SIMD_LEVEL_H
#endif
