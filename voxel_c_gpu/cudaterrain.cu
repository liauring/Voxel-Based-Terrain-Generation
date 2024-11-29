#include <cuda_runtime.h>
#include "cudaterrain.h"

__global__ void horlineHiddenKernel(
    uint32_t* screen,
    float* hidden,
    const uint8_t* heightmap,
    const uint32_t* colormap,
    float p1_x, float p1_y,
    float dx, float dy,
    float offset, float scale, float horizon,
    int width, int height, int map_size)
{
    // Each thread handles one vertical line
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;

    // Calculate position in heightmap/colormap
    float cur_x = p1_x + dx * i;
    float cur_y = p1_y + dy * i;
    
    int xi = ((int)floor(cur_x)) & (map_size - 1);
    int yi = ((int)floor(cur_y)) & (map_size - 1);

    // Calculate height and get color
    float heightonscreen = (heightmap[yi * map_size + xi] + offset) * scale + horizon;
    uint32_t color = colormap[yi * map_size + xi];

    // Draw vertical line
    int ytop = (int)heightonscreen;
    int ybottom = (int)hidden[i];
    
    if (ytop < 0) ytop = 0;
    if (ybottom > height) ybottom = height;
    
    // Draw pixels for this vertical line
    for (int y = ytop; y < ybottom; y++) {
        screen[y * width + i] = color;
    }

    // Update hidden buffer
    if (heightonscreen < hidden[i]) {
        hidden[i] = heightonscreen;
    }
}

// Host function to launch the kernel
void cudaHorlineHidden(
    uint32_t* d_screen,
    float* d_hidden,
    const uint8_t* d_heightmap,
    const uint32_t* d_colormap,
    CustomPoint p1, CustomPoint p2,
    float offset, float scale, float horizon)
{
    float dx = (p2.x - p1.x) / WIDTH;
    float dy = (p2.y - p1.y) / WIDTH;

    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocks = (WIDTH + threadsPerBlock - 1) / threadsPerBlock;

    horlineHiddenKernel<<<blocks, threadsPerBlock>>>(
        d_screen, d_hidden, d_heightmap, d_colormap,
        p1.x, p1.y, dx, dy,
        offset, scale, horizon,
        WIDTH, HEIGHT, MAP_SIZE
    );
    

}

// Memory initialization
void initCudaMemory(
    uint32_t** d_screen,
    float** d_hidden,
    uint8_t** d_heightmap,
    uint32_t** d_colormap)
{
    // Allocate device memory
    cudaCheckError(cudaMalloc(d_screen, WIDTH * HEIGHT * sizeof(uint32_t)));
    cudaCheckError(cudaMalloc(d_hidden, WIDTH * sizeof(float)));
    cudaCheckError(cudaMalloc(d_heightmap, MAP_SIZE * MAP_SIZE * sizeof(uint8_t)));
    cudaCheckError(cudaMalloc(d_colormap, MAP_SIZE * MAP_SIZE * sizeof(uint32_t)));

    // Initialize hidden buffer with HEIGHT
    float init_value = HEIGHT;
    cudaCheckError(cudaMemset(*d_hidden, init_value, WIDTH * sizeof(float)));
    
    // Initialize screen with sky color (optional)
    uint32_t sky_color = 0x87CEEB;
    cudaCheckError(cudaMemset(*d_screen, sky_color, WIDTH * HEIGHT * sizeof(uint32_t)));
}

// Copy data to GPU
void copyDataToGPU(
    uint8_t* d_heightmap, const uint8_t heightmap[MAP_SIZE][MAP_SIZE],
    uint32_t* d_colormap, const uint32_t colormap[MAP_SIZE][MAP_SIZE])
{
    // Copy heightmap to GPU
    cudaCheckError(cudaMemcpy(d_heightmap, 
                             heightmap, 
                             MAP_SIZE * MAP_SIZE * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));

    // Copy colormap to GPU
    cudaCheckError(cudaMemcpy(d_colormap, 
                             colormap, 
                             MAP_SIZE * MAP_SIZE * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
}

// Free CUDA memory
void freeCudaMemory(
    uint32_t* d_screen,
    float* d_hidden,
    uint8_t* d_heightmap,
    uint32_t* d_colormap)
{
    if (d_screen) cudaCheckError(cudaFree(d_screen));
    if (d_hidden) cudaCheckError(cudaFree(d_hidden));
    if (d_heightmap) cudaCheckError(cudaFree(d_heightmap));
    if (d_colormap) cudaCheckError(cudaFree(d_colormap));
}

