#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <cstdint>
#include <png.h>
#include <mutex>

// 定義畫布尺寸
#define WIDTH 1024
#define HEIGHT 768
#define MAP_SIZE 1024
#define NUM_BLOCKS_X 4
#define NUM_BLOCKS_Y 4

typedef struct {
    float height;
    uint32_t color;
} TerrainPixel;

// 存放渲染結果的螢幕緩衝區
uint32_t screen[WIDTH * HEIGHT];
std::mutex screen_mutex;

// 地圖數據
uint8_t heightmap[MAP_SIZE][MAP_SIZE];
uint32_t colormap[MAP_SIZE][MAP_SIZE];

// Function Prototypes
void LoadMaps(const char* heightFile, const char* colorFile);
void RenderBlock(int blockX, int blockY, int blockWidth, int blockHeight);
void DrawVerticalLine(int x, int yStart, int yEnd, uint32_t color);
void SaveToImage(const char* filename);

void LoadPNG(const char* filename, uint8_t** image, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        throw std::runtime_error("Error: Unable to open file.");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);

    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        throw std::runtime_error("Error during PNG read.");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    int rowbytes = png_get_rowbytes(png, info);

    *image = new uint8_t[*height * rowbytes];
    png_bytep rows[*height];
    for (int y = 0; y < *height; y++) {
        rows[y] = *image + y * rowbytes;
    }

    png_read_image(png, rows);
    fclose(fp);
    png_destroy_read_struct(&png, &info, nullptr);
}

void LoadMaps(const char* heightFile, const char* colorFile) {
    uint8_t* heightImage;
    uint8_t* colorImage;
    int heightWidth, heightHeight, colorWidth, colorHeight;

    LoadPNG(heightFile, &heightImage, &heightWidth, &heightHeight);
    LoadPNG(colorFile, &colorImage, &colorWidth, &colorHeight);

    if (heightWidth != MAP_SIZE || heightHeight != MAP_SIZE ||
        colorWidth != MAP_SIZE || colorHeight != MAP_SIZE) {
        throw std::runtime_error("Error: Map dimensions must match MAP_SIZE.");
    }

    // 載入高度和顏色數據
    for (int y = 0; y < MAP_SIZE; y++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            heightmap[y][x] = heightImage[y * MAP_SIZE + x];
            colormap[y][x] = (colorImage[(y * MAP_SIZE + x) * 3] << 16) |
                             (colorImage[(y * MAP_SIZE + x) * 3 + 1] << 8) |
                             colorImage[(y * MAP_SIZE + x) * 3 + 2];
        }
    }

    delete[] heightImage;
    delete[] colorImage;
}

void RenderBlock(int blockX, int blockY, int blockWidth, int blockHeight) {
    for (int y = blockY; y < blockY + blockHeight; y++) {
        for (int x = blockX; x < blockX + blockWidth; x++) {
            // 計算地圖對應像素
            int mapX = (x * MAP_SIZE) / WIDTH;
            int mapY = (y * MAP_SIZE) / HEIGHT;

            float height = heightmap[mapY][mapX];
            uint32_t color = colormap[mapY][mapX];

            // 渲染像素
            std::lock_guard<std::mutex> lock(screen_mutex);
            screen[y * WIDTH + x] = color;
        }
    }
}

int main() {
    // 載入地圖
    LoadMaps("../D7.png", "../C7W.png");

    // 設定分塊渲染參數
    int blockWidth = WIDTH / NUM_BLOCKS_X;
    int blockHeight = HEIGHT / NUM_BLOCKS_Y;

    // 創建執行緒
    std::vector<std::thread> threads;
    for (int by = 0; by < NUM_BLOCKS_Y; by++) {
        for (int bx = 0; bx < NUM_BLOCKS_X; bx++) {
            threads.emplace_back(RenderBlock, bx * blockWidth, by * blockHeight, blockWidth, blockHeight);
        }
    }

    // 等待所有執行緒完成
    for (auto& t : threads) {
        t.join();
    }

    // 儲存渲染結果
    SaveToImage("output.png");

    return 0;
}

void SaveToImage(const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        throw std::runtime_error("Error: Unable to open file for writing.");
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);

    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        throw std::runtime_error("Error during PNG write.");
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, WIDTH, HEIGHT, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    png_bytep row_pointers[HEIGHT];
    for (int y = 0; y < HEIGHT; y++) {
        row_pointers[y] = (png_bytep)malloc(WIDTH * 3);
        for (int x = 0; x < WIDTH; x++) {
            uint32_t color = screen[y * WIDTH + x];
            row_pointers[y][x * 3] = (color >> 16) & 0xFF;      // R
            row_pointers[y][x * 3 + 1] = (color >> 8) & 0xFF;  // G
            row_pointers[y][x * 3 + 2] = color & 0xFF;         // B
        }
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    for (int y = 0; y < HEIGHT; y++) {
        free(row_pointers[y]);
    }

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}