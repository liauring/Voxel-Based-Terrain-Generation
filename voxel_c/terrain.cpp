#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define WIDTH 700
#define HEIGHT 512
#define MAP_SIZE 1024

using namespace cv;

// Structures
typedef struct {
    float x, y;
} CustomPoint;  // Renamed from Point

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

int main() {
    Init("./C7W.png", "./D7.png");

    for (int i = 0; i < 64; i++) {
        printf("Rendering Frame %d\n", i);
        DrawFrontToBack((CustomPoint){(float)670, (float)(500 - i * 16)}, 0, 120, 10000, (CustomPoint){(float)670, (float)(500 - i * 16)});
        SaveFrameAsImage(i); // Save the current frame as an image
    }

    return 0;
}

// -----------------------------------------------------

void Init(const char *colorfilename, const char *heightfilename) {
    // Load color image
    Mat colorImage = imread(colorfilename, IMREAD_COLOR);
    if (colorImage.empty()) {
        fprintf(stderr, "Error: Unable to load color file: %s\n", colorfilename);
        exit(1);
    }

    // Load height image
    Mat heightImage = imread(heightfilename, IMREAD_GRAYSCALE);
    if (heightImage.empty()) {
        fprintf(stderr, "Error: Unable to load height file: %s\n", heightfilename);
        exit(1);
    }

    // Verify dimensions
    if (colorImage.cols != MAP_SIZE || colorImage.rows != MAP_SIZE ||
        heightImage.cols != MAP_SIZE || heightImage.rows != MAP_SIZE) {
        fprintf(stderr, "Error: Maps must be %dx%d in size.\n", MAP_SIZE, MAP_SIZE);
        exit(1);
    }

    // Populate colormap and heightmap
    for (int y = 0; y < MAP_SIZE; y++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            Vec3b color = colorImage.at<Vec3b>(y, x);
            uint8_t height = heightImage.at<uint8_t>(y, x);

            colormap[x][y] = (color[2] << 16) | (color[1] << 8) | color[0]; // BGR to RGB
            heightmap[x][y] = height;
        }
    }

    ClearScreen(0xffa366); // Initial screen clear color
}

// -----------------------------------------------------

void ClearScreen(uint32_t color) {
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        screen[i] = color;
    }
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
    int n = WIDTH;
    float dx = (p2.x - p1.x) / n;
    float dy = (p2.y - p1.y) / n;

    for (int i = 0; i < n; i++) {
        int xi = ((int)floor(p1.x)) & (MAP_SIZE - 1);
        int yi = ((int)floor(p1.y)) & (MAP_SIZE - 1);

        float heightonscreen = (heightmap[xi][yi] + offset) * scale + horizon;
        uint32_t color = colormap[xi][yi];

        DrawVerticalLine(i, (int)heightonscreen, (int)hidden[i], color);

        if (heightonscreen < hidden[i]) {
            hidden[i] = heightonscreen;
        }

        p1.x += dx;
        p1.y += dy;
    }
}

// -----------------------------------------------------

void DrawFrontToBack(CustomPoint p, float phi, float height, float distance, CustomPoint pmap) {
    ClearScreen(0xffa366); // Reset the screen

    for (int i = 0; i < WIDTH; i++) {
        hidden[i] = HEIGHT;
    }

    float dz = 1.0f;
    for (float z = 5; z < distance; z += dz) {
        CustomPoint pl = {-z, -z};
        CustomPoint pr = { z, -z};

        pl = Rotate(pl, phi);
        pr = Rotate(pr, phi);

        HorlineHidden(
            (CustomPoint){p.x + pl.x, p.y + pl.y},
            (CustomPoint){p.x + pr.x, p.y + pr.y},
            -height, -1.0f / z * 240.0f, 100, pmap);

        dz += 0.1f; // Increment dz gradually for depth
    }
}

// -----------------------------------------------------

void SaveFrameAsImage(int frameNumber) {
    Mat frameImage(HEIGHT, WIDTH, CV_8UC3);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            uint32_t color = screen[y * WIDTH + x];
            uint8_t r = (color >> 16) & 0xFF;
            uint8_t g = (color >> 8) & 0xFF;
            uint8_t b = color & 0xFF;

            frameImage.at<Vec3b>(y, x) = Vec3b(b, g, r); // Convert RGB to BGR
        }
    }

    char filename[64];
    snprintf(filename, sizeof(filename), "./output/frame_%03d.png", frameNumber);
    imwrite(filename, frameImage);
}


