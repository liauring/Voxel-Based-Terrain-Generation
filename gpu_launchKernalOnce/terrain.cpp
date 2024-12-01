#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <png.h>
#include <sys/stat.h> // For mkdir
#include <sys/types.h> // For mode_t
#include <iostream>
#include <chrono>
#include <omp.h>
#include "cudaterrain.h"

#define WIDTH 700
#define HEIGHT 512
#define MAP_SIZE 1024
#define DZ 0.1f
#define DZ_STEP 0.000001f



// Globals
uint32_t screen[WIDTH * HEIGHT];
uint8_t heightmap[MAP_SIZE][MAP_SIZE];
uint32_t colormap[MAP_SIZE][MAP_SIZE];
float hidden[WIDTH];

// Function Prototypes
void Init(const char *colorfilename, const char *heightfilename);
void DrawVerticalLine(int x, int ytop, int ybottom, uint32_t color);
void HorlineHidden(CustomPoint p1, CustomPoint p2, float offset, float scale, float horizon, CustomPoint pmap);
CustomPoint Rotate(CustomPoint p, float phi);
void ClearScreen(uint32_t color);
void DrawFrontToBack(CustomPoint p, float phi, float height, float distance, CustomPoint pmap);
void SaveFrameAsImage(int frameNumber);
void CreateOutputFolder(const char *foldername);

// Add GPU memory pointers with other globals
uint32_t* d_screen;
float* d_hidden;
uint8_t* d_heightmap;
uint32_t* d_colormap;

int main() {
    

    // Ensure output folder exists
    CreateOutputFolder("./output");
    
    Init("../C7W.png", "../D7.png");
    // Initialize CUDA memory
    initCudaMemory(&d_screen, &d_hidden, &d_heightmap, &d_colormap);
    copyDataToGPU(d_heightmap, heightmap, d_colormap, colormap);
    const auto compute_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 64; i++) {
        printf("Rendering Frame %d\n", i);
        DrawFrontToBack((CustomPoint){(float)670, (float)(500 - i * 16)}, 0, 120, 10000, (CustomPoint){(float)670, (float)(500 - i * 16)});
        // SaveFrameAsImage(i); // Save the current frame as an image
    }
    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    // Cleanup CUDA memory
    freeCudaMemory(d_screen, d_hidden, d_heightmap, d_colormap);
    
    return 0;
}

// -----------------------------------------------------

void CreateOutputFolder(const char *foldername) {
    struct stat st = {0};

    // Check if folder exists
    if (stat(foldername, &st) == -1) {
        // Create folder
        if (mkdir(foldername, 0755) != 0) {
            fprintf(stderr, "Error: Unable to create folder: %s\n", foldername);
            exit(1);
        }
    }
}


void ReadPNG(const char *filename, uint8_t **image, int *width, int *height, int *channels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open file: %s\n", filename);
        exit(1);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Error: Unable to create PNG read structure.\n");
        fclose(fp);
        exit(1);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Error: Unable to create PNG info structure.\n");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        exit(1);
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error: During PNG read operation.\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        exit(1);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    *channels = png_get_channels(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (png_get_color_type(png, info) == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (png_get_color_type(png, info) == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_color_type(png, info) & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);

    png_read_update_info(png, info);

    size_t rowbytes = png_get_rowbytes(png, info);
    *image = (uint8_t *)malloc(*height * rowbytes);

    png_bytep rows[*height];
    for (int i = 0; i < *height; i++) rows[i] = *image + i * rowbytes;

    png_read_image(png, rows);

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
}

void Init(const char *colorfilename, const char *heightfilename) {
    uint8_t *colorImage = NULL, *heightImage = NULL;
    int colorWidth, colorHeight, heightWidth, heightHeight, colorChannels, heightChannels;

    // Read color PNG
    ReadPNG(colorfilename, &colorImage, &colorWidth, &colorHeight, &colorChannels);
    if (colorWidth != MAP_SIZE || colorHeight != MAP_SIZE) {
        fprintf(stderr, "Error: Color map must be %dx%d.\n", MAP_SIZE, MAP_SIZE);
        free(colorImage);
        exit(1);
    }

    // Read height PNG
    ReadPNG(heightfilename, &heightImage, &heightWidth, &heightHeight, &heightChannels);
    if (heightWidth != MAP_SIZE || heightHeight != MAP_SIZE) {
        fprintf(stderr, "Error: Height map must be %dx%d.\n", MAP_SIZE, MAP_SIZE);
        free(colorImage);
        free(heightImage);
        exit(1);
    }

    // Populate colormap and heightmap
    for (int y = 0; y < MAP_SIZE; y++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            int idx = (y * MAP_SIZE + x) * 3;
            colormap[x][y] = (colorImage[idx] << 16) | (colorImage[idx + 1] << 8) | colorImage[idx + 2];
            heightmap[x][y] = heightImage[y * MAP_SIZE + x];
        }
    }

    free(colorImage);
    free(heightImage);
    ClearScreen(0x87CEEB); // Clear the screen
}
// -----------------------------------------------------

void ClearScreen(uint32_t color) {
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        screen[i] = color;
    }
    // Clear GPU screen
    //cudaMemset(d_screen, color, WIDTH * HEIGHT * sizeof(uint32_t));
    cudaMemcpy(d_screen, screen, WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyHostToDevice);

    for (int i = 0; i < WIDTH; i++) {
        hidden[i] = HEIGHT;
    }
    // Reset hidden buffer on GPU
    //float init_hidden = HEIGHT;
    //cudaMemset(d_hidden, init_hidden, WIDTH * sizeof(float));
    cudaMemcpy(d_hidden, hidden, WIDTH * sizeof(float), cudaMemcpyHostToDevice);

}

// -----------------------------------------------------

void DrawVerticalLine(int x, int ytop, int ybottom, uint32_t color) {
    if (ytop >= ybottom) return;
    if (ytop < 0) ytop = 0;
    if (ybottom > HEIGHT) ybottom = HEIGHT;

    for (int y = ytop; y < ybottom; y++) {
        screen[y * WIDTH + x] = color;
    }
}

// -----------------------------------------------------

CustomPoint Rotate(CustomPoint p, float phi) {
    float xtemp = p.x * cos(phi) + p.y * sin(phi);
    float ytemp = p.x * -sin(phi) + p.y * cos(phi);
    return (CustomPoint){xtemp, ytemp};
}

// -----------------------------------------------------

void HorlineHidden(CustomPoint p1, CustomPoint p2, float offset, float scale, float horizon, CustomPoint pmap) {
    // cudaHorlineHidden(
    //     d_screen,
    //     d_hidden,
    //     d_heightmap,
    //     d_colormap,
    //     p1, p2,
    //     offset, scale, horizon
    // );
}

// -----------------------------------------------------

void DrawFrontToBack(CustomPoint p, float phi, float height, float distance, CustomPoint pmap) {
    //ClearScreen(0x87CEEB);

    // Launch kernel through wrapper function
    launchRenderKernel(
        d_screen,
        d_hidden,
        d_heightmap,
        d_colormap,
        p.x, p.y,
        phi,
        height,
        distance
    );
    
    // Copy result back
    cudaMemcpy(screen, d_screen, WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

// -----------------------------------------------------

void SaveFrameAsImage(int frameNumber) {
    char filename[64];
    snprintf(filename, sizeof(filename), "./output/frame_%03d.png", frameNumber);

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open file for writing: %s\n", filename);
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Error: Unable to create PNG write structure.\n");
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Error: Unable to create PNG info structure.\n");
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error: During PNG write operation.\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    png_set_IHDR(png, info, WIDTH, HEIGHT, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    png_bytep row_pointers[HEIGHT];
    for (int y = 0; y < HEIGHT; y++) {
        row_pointers[y] = (png_bytep)malloc(WIDTH * 3);
        for (int x = 0; x < WIDTH; x++) {
            uint32_t color = screen[y * WIDTH + x];
            row_pointers[y][x * 3] = (color >> 16) & 0xFF;      // Red
            row_pointers[y][x * 3 + 1] = (color >> 8) & 0xFF;  // Green
            row_pointers[y][x * 3 + 2] = color & 0xFF;         // Blue
        }
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    for (int y = 0; y < HEIGHT; y++) {
        free(row_pointers[y]);
    }

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}


