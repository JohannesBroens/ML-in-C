# Makefile for CUDA MLP example

CUDA_PATH ?= /usr/lib/cuda
NVCC := /usr/bin/nvcc

TARGET = mlp
SRC = mlp.cu
OBJ = $(SRC:.cu=.o)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^ 

clean:
	rm -f $(TARGET) $(OBJ)
