# Compiler and Flags
CC = gcc
CFLAGS = -Wall -O2 -std=c++17
LDFLAGS = -lpng -lm

# Target and Source Files
TARGET = terrain
SRC = terrain.cpp

# Default Target
all: $(TARGET)

# Compile the Program
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean: Remove Executable and Output Folder
clean:
	rm -f $(TARGET)
	rm -rf output