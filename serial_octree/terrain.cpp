#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>

#define WIDTH 1024
#define HEIGHT 768
#define MAP_SIZE 256
#define MAX_DEPTH 6

// 顏色與高度資料
uint8_t heightmap[MAP_SIZE][MAP_SIZE][MAP_SIZE];
uint32_t colormap[MAP_SIZE][MAP_SIZE][MAP_SIZE];
uint32_t screen[WIDTH * HEIGHT];

// 八叉樹節點結構
struct OctreeNode {
    int x, y, z;      // 節點起始位置
    int size;         // 節點大小
    bool isLeaf;      // 是否為葉節點
    uint32_t color;   // 如果是葉節點，則存儲顏色
    OctreeNode* children[8]; // 八個子節點指標

    OctreeNode(int x, int y, int z, int size)
        : x(x), y(y), z(z), size(size), isLeaf(false), color(0) {
        for (int i = 0; i < 8; i++) children[i] = nullptr;
    }

    ~OctreeNode() {
        for (int i = 0; i < 8; i++) {
            if (children[i]) delete children[i];
        }
    }
};

// 八叉樹根節點
OctreeNode* root;

// 檢查是否需要分裂節點
bool ShouldSplit(int x, int y, int z, int size) {
    uint32_t firstColor = colormap[x][y][z];
    uint8_t firstHeight = heightmap[x][y][z];
    for (int i = x; i < x + size; i++) {
        for (int j = y; j < y + size; j++) {
            for (int k = z; k < z + size; k++) {
                if (colormap[i][j][k] != firstColor || heightmap[i][j][k] != firstHeight) {
                    return true;
                }
            }
        }
    }
    return false;
}

// 建立八叉樹
OctreeNode* BuildOctree(int x, int y, int z, int size) {
    OctreeNode* node = new OctreeNode(x, y, z, size);

    // 如果節點太小，或者所有體素顏色與高度相同，則不再分裂
    if (size == 1 || !ShouldSplit(x, y, z, size)) {
        node->isLeaf = true;
        node->color = colormap[x][y][z];
        return node;
    }

    // 分裂為 8 個子節點
    int half = size / 2;
    node->children[0] = BuildOctree(x, y, z, half);
    node->children[1] = BuildOctree(x + half, y, z, half);
    node->children[2] = BuildOctree(x, y + half, z, half);
    node->children[3] = BuildOctree(x + half, y + half, z, half);
    node->children[4] = BuildOctree(x, y, z + half, half);
    node->children[5] = BuildOctree(x + half, y, z + half, half);
    node->children[6] = BuildOctree(x, y + half, z + half, half);
    node->children[7] = BuildOctree(x + half, y + half, z + half, half);

    return node;
}


void RenderOctree(OctreeNode* node, float observerX, float observerY, float observerZ) {
    if (!node) return;

    // 如果是葉節點，直接渲染
    if (node->isLeaf) {
        int screenX = (node->x * WIDTH) / MAP_SIZE;
        int screenY = (node->y * HEIGHT) / MAP_SIZE;

        // 計算深度遮擋
        int z = node->z;
        int screenIndex = screenY * WIDTH + screenX;
        if (z < screen[screenIndex]) {
            screen[screenIndex] = node->color;
        }
        return;
    }

    // 非葉節點，遞迴渲染子節點
    for (int i = 0; i < 8; i++) {
        RenderOctree(node->children[i], observerX, observerY, observerZ);
    }
}

int main() {
    // 加載數據
    // heightmap 和 colormap 需要初始化，假設其已填充為 MAP_SIZE^3 的數據。

    // 構建八叉樹
    root = BuildOctree(0, 0, 0, MAP_SIZE);

    // 初始化螢幕
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        screen[i] = 0x87CEEB; // 預設為藍色背景
    }

    // 渲染
    RenderOctree(root, 128.0f, 128.0f, 128.0f); // 假設觀察者在中心

    // 儲存圖像
    SaveToImage("output.png");

    // 清理八叉樹
    delete root;

    return 0;
}