#ifndef CUDATERRAIN_H
#define CUDATERRAIN_H

#include <cuda_runtime.h>
#include <stdint.h>

// Constants (matching terrain.cpp)
#define WIDTH 700
#define HEIGHT 512
#define MAP_SIZE 1024

// Structure definition (matching terrain.cpp)
typedef struct {
    float x, y;
} CustomPoint;

// CUDA function declarations
void cudaHorlineHidden(
    uint32_t* d_screen,
    float* d_hidden,
    const uint8_t* d_heightmap,
    const uint32_t* d_colormap,
    CustomPoint p1, CustomPoint p2,
    float offset, float scale, float horizon
);

// Memory management functions
void initCudaMemory(
    uint32_t** d_screen,
    float** d_hidden,
    uint8_t** d_heightmap,
    uint32_t** d_colormap
);

void copyDataToGPU(
    uint8_t* d_heightmap, const uint8_t heightmap[][MAP_SIZE],
    uint32_t* d_colormap, const uint32_t colormap[][MAP_SIZE]
);

void freeCudaMemory(
    uint32_t* d_screen,
    float* d_hidden,
    uint8_t* d_heightmap,
    uint32_t* d_colormap
);

// Error checking helper
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
    //   fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif // CUDATERRAIN_H


