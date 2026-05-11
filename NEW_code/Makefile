NVCC := nvcc

CUDA_ARCH := -arch=sm_75
NVCC_FLAGS := $(CUDA_ARCH) -std=c++17 -O2 -Iinclude

BUILD_DIR := build

KERNEL_SRCS := src/kernels/gemm.cu src/kernels/attention.cu src/kernels/layernorm.cu src/kernels/activations.cu src/kernels/embedding.cu
MODEL_SRCS := src/model/transformer.cu
DATA_SRCS := src/data/dataset.cpp
MAIN_SRC := src/main.cu

KERNEL_OBJS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(KERNEL_SRCS))
MODEL_OBJS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(MODEL_SRCS))
DATA_OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(DATA_SRCS))
MAIN_OBJ := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(MAIN_SRC))

TARGET := $(BUILD_DIR)/train
TEST_GEMM := $(BUILD_DIR)/test_gemm
TEST_ATTN := $(BUILD_DIR)/test_attention
TEST_LN := $(BUILD_DIR)/test_layernorm
TEST_MODEL := $(BUILD_DIR)/test_model_reference
TEST_FUSION := $(BUILD_DIR)/test_fusion_benchmark
TEST_EMBEDDING := $(BUILD_DIR)/test_embedding
TEST_SOFTMAX_CE := $(BUILD_DIR)/test_softmax_ce

.PHONY: all tests clean

all: $(TARGET)

tests: $(TEST_GEMM) $(TEST_ATTN) $(TEST_LN) $(TEST_MODEL) $(TEST_FUSION) $(TEST_EMBEDDING) $(TEST_SOFTMAX_CE)

$(TARGET): $(KERNEL_OBJS) $(MODEL_OBJS) $(DATA_OBJS) $(MAIN_OBJ) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_GEMM): tests/test_gemm.cu $(KERNEL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TEST_ATTN): tests/test_attention.cu $(KERNEL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TEST_LN): tests/test_layernorm.cu $(KERNEL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TEST_MODEL): tests/test_model_reference.cu $(KERNEL_OBJS) $(MODEL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TEST_FUSION): tests/test_fusion_benchmark.cu $(KERNEL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TEST_EMBEDDING): tests/test_embedding.cu $(KERNEL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TEST_SOFTMAX_CE): tests/test_softmax_ce.cu $(KERNEL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -rf $(BUILD_DIR)
