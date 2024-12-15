# Compiler and Flags
NVCC = nvcc
CC = g++
NVCCFLAGS = -O2 -arch=sm_60 -Xcompiler=-fopenmp
CFLAGS = -Wall -O2 -std=c++17 -fopenmp
LDFLAGS = -lpng -lm

# Directories
SRC_DIR = .
BUILD_DIR = build
OUTPUT_DIR = output

# Target and Source Files
TARGET = terrain
CUDA_SRC = cudaterrain.cu
CPP_SRC = terrain.cpp
HEADERS = cudaterrain.h

# Object files
CUDA_OBJ = $(BUILD_DIR)/cudaterrain.o
CPP_OBJ = $(BUILD_DIR)/terrain.o

# Create directories
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(OUTPUT_DIR))

# Default Target
all: $(TARGET)

# Link everything together
$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# Compile CUDA source
$(CUDA_OBJ): $(CUDA_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source
$(CPP_OBJ): $(CPP_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
	rm -rf $(OUTPUT_DIR)

.PHONY: all clean