#if defined(_WIN32) || defined(_WIN64)
#   ifndef FASTNOISE_EXPORT
#     ifdef fastNoise_EXPORTS
#       define FASTNOISE_EXPORT __declspec(dllexport)
#     else
#       define FASTNOISE_EXPORT __declspec(dllimport)
#     endif
#   endif
#else
#   define FASTNOISE_EXPORT 
#endif

#ifndef FASTNOISE_NO_EXPORT
#  define FASTNOISE_NO_EXPORT 
#endif
