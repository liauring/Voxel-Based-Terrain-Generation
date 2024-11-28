# Compiler and Flags
NVCC = nvcc
CC = g++
NVCCFLAGS = -O2 -arch=sm_60 -Xcompiler=-fopenmp
CFLAGS = -Wall -O2 -std=c++17 -fopenmp
LDFLAGS = -lpng -lm

# Target and Source Files
TARGET = terrain
CUDA_SRC = cudaterrain.cu
CPP_SRC = terrain.cpp

# Object files
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
CPP_OBJ = $(CPP_SRC:.cpp=.o)

# Default Target
all: $(TARGET)

# Link everything together
$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# Compile CUDA source
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean: Remove Executable, Object Files, and Output Folder
clean:
	rm -f $(TARGET) *.o
	rm -rf output