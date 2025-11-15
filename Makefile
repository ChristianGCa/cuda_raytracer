NVCC = nvcc
SRCS = main.cc cuda_renderer.cu raytracing_cuda.cu
TARGET = raytracer
BUILD_DIR = build
BIN = $(BUILD_DIR)/$(TARGET)

.PHONY: all run clean

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN): $(SRCS) | $(BUILD_DIR)
	$(NVCC) -std=c++14 -O3 $(SRCS) -lcurand -o $(BIN)

run: all
	$(BIN) > $(BUILD_DIR)/out.ppm

clean:
	rm -rf $(BUILD_DIR)
