#include <cuda_runtime.h>
#include "cudaterrain.h"
#include <iostream>
#include <chrono>



__device__ CustomPoint deviceRotate(CustomPoint p, float phi) {
    float xtemp = p.x * cos(phi) + p.y * sin(phi);
    float ytemp = p.x * -sin(phi) + p.y * cos(phi);
    return (CustomPoint){xtemp, ytemp};
}


__global__ void CalculateDepthsKernel(
    DepthPixel* depth_buffer,  // WIDTH * num_depths size
    const uint8_t* heightmap,
    const uint32_t* colormap,
    float p_x, float p_y,
    float phi,
    float height,
    int num_depths)  // Current z value
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= WIDTH || blockIdx.y * blockDim.y + threadIdx.y >= num_depths) return;
    float z = 5.0f + blockIdx.y * 0.1f;

    float scale = -1.0f / z * 240.0f;
    float offset = -height;
    float horizon = 100;

    // Calculate points for this z
    CustomPoint pl = {-z, -z};
    CustomPoint pr = {z, -z};

    pl = deviceRotate(pl, phi);
    pr = deviceRotate(pr, phi);

    CustomPoint p1 = {p_x + pl.x, p_y + pl.y};
    CustomPoint p2 = {p_x + pr.x, p_y + pr.y};

    float dx = (p2.x - p1.x) / WIDTH;
    float dy = (p2.y - p1.y) / WIDTH;

    float cur_x = p1.x + dx * x;
    float cur_y = p1.y + dy * x;
    
    int xi = ((int)floor(cur_x)) & (MAP_SIZE - 1);
    int yi = ((int)floor(cur_y)) & (MAP_SIZE - 1);

    float heightonscreen = (heightmap[yi * MAP_SIZE + xi] + offset) * scale + horizon;
    
    int depth_idx = (blockIdx.y * blockDim.y + threadIdx.y)* WIDTH + x;
    depth_buffer[depth_idx].point_height = heightonscreen;
    depth_buffer[depth_idx].color = colormap[yi * MAP_SIZE + xi];
}


__global__ void MergeDepthsKernel(
    uint32_t* screen,
    DepthPixel* depth_buffer,
    int num_depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Declare shared memory for hidden buffer
    extern __shared__ float shared_hidden[];
    if (x >= WIDTH) return;
    for(int y = 0; y < HEIGHT; y++) {
        screen[y * WIDTH + x] = BLUE;
    }
    shared_hidden[threadIdx.x] = HEIGHT;
    
    //Process all depths for this x coordinate
    for (int d = 0; d < num_depths; d++) {
        DepthPixel pixel = depth_buffer[d * WIDTH + x];
        //if (!pixel.valid) continue;
        
        if (pixel.point_height < shared_hidden[threadIdx.x]) {
            int ytop = (int)pixel.point_height;
            int ybottom = (int)shared_hidden[threadIdx.x];
            
            if (ytop < 0) ytop = 0;
            if (ybottom > HEIGHT) ybottom = HEIGHT;
            
            for (int y = ytop; y < ybottom; y++) {
                screen[y * WIDTH + x] = pixel.color;
            }
            shared_hidden[threadIdx.x] = pixel.point_height;
        }
    }
}

void launchRenderKernel(
    uint32_t* d_screen,
    const uint8_t* d_heightmap,
    const uint32_t* d_colormap,
    float p_x, float p_y,
    float phi,
    float height,
    float distance)
{
    // Calculate number of z steps
    int num_depths = (int)((distance - 5.0f) / DZ);
    
    // Allocate depth buffer
    //Can be moved to initCudaMemory
    DepthPixel* d_depth_buffer;
    cudaCheckError(cudaMalloc(&d_depth_buffer, WIDTH * num_depths * sizeof(DepthPixel)));
    
    // Launch CalculateDepthsKernel
    // All depths are calculated in parallel
    dim3 threadsPerBlock(256, 2);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (num_depths + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    CalculateDepthsKernel<<<numBlocks, threadsPerBlock>>>(
        d_depth_buffer, d_heightmap, d_colormap, p_x, p_y, phi, height, num_depths); 

    cudaDeviceSynchronize();
    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time for CalculateDepthsKernel: " << elapsed.count() << " s\n";
    // Launch merge kernel
    int mergeThreads = 256;
    int mergeBlocks = (WIDTH + mergeThreads - 1) / mergeThreads;
    size_t sharedMemSize = mergeThreads * sizeof(float);  // Allocate shared memory for hidden buffer
    // start timer
    start = std::chrono::high_resolution_clock::now();
    MergeDepthsKernel<<<mergeBlocks, mergeThreads, sharedMemSize>>>(
        d_screen, d_depth_buffer, num_depths);
    cudaDeviceSynchronize();
    // end timer
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Elapsed time for MergeDepthsKernel: " << elapsed.count() << " s\n";
    // can be freed after all screen are rendered
    cudaFree(d_depth_buffer);
}



// Memory initialization
void initCudaMemory(
    uint32_t** d_screen,
    uint8_t** d_heightmap,
    uint32_t** d_colormap)
{
    // Allocate device memory
    cudaCheckError(cudaMalloc(d_screen, WIDTH * HEIGHT * sizeof(uint32_t)));
    cudaCheckError(cudaMalloc(d_heightmap, MAP_SIZE * MAP_SIZE * sizeof(uint8_t)));
    cudaCheckError(cudaMalloc(d_colormap, MAP_SIZE * MAP_SIZE * sizeof(uint32_t)));

    
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
    uint8_t* d_heightmap,
    uint32_t* d_colormap)
{
    if (d_screen) cudaCheckError(cudaFree(d_screen));
    if (d_heightmap) cudaCheckError(cudaFree(d_heightmap));
    if (d_colormap) cudaCheckError(cudaFree(d_colormap));
}

