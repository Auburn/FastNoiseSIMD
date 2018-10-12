#include "FastNoiseSIMD/FastNoiseSIMD.h"

#ifdef _MSC_VER
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <realtimeapiset.h>
#endif

#if FN_USE_FILESYSTEM == 1
#include <filesystem>
namespace fs=std::filesystem;
#elif FN_USE_FILESYSTEM == 2
#include <experimental/filesystem>
#ifdef _MSC_VER
namespace fs=std::experimental::filesystem::v1;
#else
namespace fs=std::experimental::filesystem;
#endif
#endif

#include <iostream>
#include <cassert>

struct NoiseInfo
{
    NoiseInfo(std::string name, FastNoise::NoiseType type):name(name), type(type) {}

    std::string name;
    FastNoise::NoiseType type;
};

//enum NoiseType { Value, ValueFractal, Perlin, PerlinFractal, Simplex, SimplexFractal, WhiteNoise, Cellular, Cubic, CubicFractal };
std::vector<NoiseInfo> Noises=
{
    {"Value", FastNoise::NoiseType::Value},
    {"ValueFractal", FastNoise::NoiseType::ValueFractal},
    {"Perlin", FastNoise::NoiseType::Perlin},
    {"PerlinFractal", FastNoise::NoiseType::PerlinFractal},
    {"Simplex", FastNoise::NoiseType::Simplex},
    {"SimplexFractal", FastNoise::NoiseType::SimplexFractal},
    {"WhiteNoise", FastNoise::NoiseType::WhiteNoise},
//    {"Cellular", FastNoise::NoiseType::Cellular},
    {"Cubic", FastNoise::NoiseType::Cubic},
    {"CubicFractal", FastNoise::NoiseType::CubicFractal}
};

std::vector<std::string> SIMDNames=
{
    {"NONE"},
    {"NEON"},
    {"SSE2"},
    {"SSE41"},
    {"AVX2"},
    {"AVX512"}
};

void saveNoise(std::string &fileName, float *data, size_t x, size_t y, size_t z)
{
    FILE *file=fopen(fileName.c_str(), "w");

    if(!file)
        return;

    size_t size=x*y*z;

    fwrite(data, sizeof(float), size, file);
    fclose(file);
}

void generate()
{
#if FN_USE_FILESYSTEM == 0
    assert(false);
#else
    int maxLevel=FastNoise::NoiseSIMD::GetSIMDLevel();
    int xSize=64;
    int ySize=64;
    int zSize=64;

//    maxLevel=2;
    fs::path dataDir("./data");

    if(!fs::exists(dataDir))
        fs::create_directory(dataDir);

    std::string fileName;
    //skip neon
    for(int i=maxLevel; i>=0; --i)
    {
        FastNoise::NoiseSIMD::SetSIMDLevel((FastNoise::SIMDType)i);
        float* noiseSet=FastNoise::NoiseSIMD::GetEmptySet(xSize, ySize, zSize);
        FastNoise::NoiseSIMD *noise=FastNoise::NoiseSIMD::New();

        for(auto &info:Noises)
        {
            noise->SetNoiseType(info.type);
            noise->FillSet(noiseSet, 0, 0, 0, xSize, ySize, zSize);

            fileName=dataDir.string()+"/"+info.name+"_"+SIMDNames[i]+".ns";
            saveNoise(fileName, noiseSet, xSize, ySize, zSize);
        }
        delete noise;
    }
#endif
}

void testPerformance()
{
    int maxLevel=FastNoise::NoiseSIMD::GetSIMDLevel();
    int xSize=64;
    int ySize=64;
    int zSize=64;

    //    maxLevel=2;
    
    std::vector<float *> noiseSets(6, nullptr);

    for(auto &info:Noises)
    {
        std::cout<<info.name<<" --------------------------------------------------------\n";

        for(int i=maxLevel; i>=0; --i)
        {
            if(!FastNoise::NoiseSIMD::SetSIMDLevel((FastNoise::SIMDType)i))
                continue;

            if(!noiseSets[i])
                noiseSets[i]=FastNoise::NoiseSIMD::GetEmptySet(xSize, ySize, zSize);

            float *noiseSet=noiseSets[i];
            FastNoise::NoiseSIMD *noise=FastNoise::NoiseSIMD::New();
            
            noise->SetNoiseType(info.type);

#ifdef _MSC_VER
            DWORD elapsed;
            DWORD start=GetTickCount();

//            ULONGLONG start;
//            ULONGLONG elapsed;
//
//            QueryInterruptTime(&start);
#endif

            for(size_t j=0; j<100; ++j)
                noise->FillSet(noiseSet, j*xSize, j*ySize, j*zSize, xSize, ySize, zSize);


#ifdef _MSC_VER
            elapsed=GetTickCount();
            elapsed=elapsed-start;

//            QueryInterruptTime(&elapsed);
//            elapsed=(elapsed-start)/10000;

            std::cout<<"    "<<SIMDNames[i]<<" "<<elapsed<<"ms\n";
#endif
            delete noise;
        }
    }

    std::cout<<"\nPress Enter\n";
    getchar();
}

int main(int argc, char ** argv)
{
    FastNoise::NoiseSIMD::loadSimd("./");
//    generate();
    testPerformance();
    return 0;
}
