# build the cuda and mpi training code.

NVCC       := nvcc
CXX        := g++
MPICXX     := mpicxx

CUDA_ARCH  := -arch=sm_75
EXTRA_NVCC_FLAGS ?=
USE_NVTX ?= 0
USE_OPENMP ?= 1
OPENMP_FLAG ?= -fopenmp

CUDA_LIB_DIR      ?= $(shell dirname $$(find /home/cme213/software/nvidia-hpc-sdk \
                     /opt/ohpc/pub/compiler/cuda/hpc_sdk \
                     -path '*/cuda/*/targets/x86_64-linux/lib/libcudart.so' \
                     2>/dev/null | head -n 1))
CUBLAS_LIB_DIR    ?= $(shell dirname $$(find /home/cme213/software/nvidia-hpc-sdk \
                     /opt/ohpc/pub/compiler/cuda/hpc_sdk \
                     -path '*/math_libs/*/targets/x86_64-linux/lib/libcublas.so' \
                     -not -path '*/stubs/*' -not -path '*/REDIST/*' \
                     2>/dev/null | head -n 1))
CUBLAS_INC_DIR    ?= $(shell realpath -m "$(CUBLAS_LIB_DIR)/../include" 2>/dev/null)
NCCL_LIB_DIR      ?= $(shell dirname $$(find /home/cme213/software/nvidia-hpc-sdk \
                     /opt/ohpc/pub/compiler/cuda/hpc_sdk \
                     -path '*/comm_libs/*/nccl/lib/libnccl.so' \
                     -not -path '*/REDIST/*' 2>/dev/null | head -n 1))
NCCL_INC_DIR      ?= $(if $(NCCL_LIB_DIR),$(shell realpath -m "$(NCCL_LIB_DIR)/../include" 2>/dev/null),)

NVTX_COMPILE := $(if $(filter 1,$(USE_NVTX)),-DCME213_USE_NVTX,)
NVTX_LINK := $(if $(filter 1,$(USE_NVTX)),-L$(CUDA_LIB_DIR) -Xlinker -rpath=$(CUDA_LIB_DIR) -lnvToolsExt,)
OPENMP_COMPILE := $(if $(filter 1,$(USE_OPENMP)),-Xcompiler $(OPENMP_FLAG),)
OPENMP_LINK := $(if $(filter 1,$(USE_OPENMP)),$(OPENMP_FLAG),)
NVCC_FLAGS := $(CUDA_ARCH) -std=c++17 -O2 -Iinclude --default-stream per-thread \
              $(if $(CUBLAS_INC_DIR),-I$(CUBLAS_INC_DIR),) \
              $(NVTX_COMPILE) $(EXTRA_NVCC_FLAGS)
CXX_FLAGS  := -std=c++17 -O2 -Iinclude

MPI_COMPILE_FLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_INCLUDE_FLAGS := $(filter -I%,$(MPI_COMPILE_FLAGS))
MPI_PTHREAD_FLAG  := $(filter -pthread,$(MPI_COMPILE_FLAGS))
MPI_LINK_FLAGS    := $(shell $(MPICXX) --showme:link 2>/dev/null)
NVCC_MPI_FLAGS    := $(NVCC_FLAGS) -DUSE_MPI $(MPI_INCLUDE_FLAGS) \
                     $(if $(MPI_PTHREAD_FLAG),-Xcompiler -pthread,) \
                     $(OPENMP_COMPILE) \
                     $(if $(NCCL_INC_DIR),-DUSE_NCCL -I$(NCCL_INC_DIR),)

KERNEL_SRCS := src/kernels/gemm.cu src/kernels/attention.cu \
               src/kernels/layernorm.cu src/kernels/activations.cu \
               src/kernels/embedding.cu
MODEL_SRCS  := src/model/transformer.cu
DATA_SRCS   := src/data/dataset.cpp
DIST_SRCS   := src/model/distributed.cpp
MAIN_SRC    := src/main.cu

ALL_CU_SRCS := $(KERNEL_SRCS) $(MODEL_SRCS) $(MAIN_SRC)
ALL_CPP_SRCS := $(DATA_SRCS) $(DIST_SRCS)

BUILD_DIR   := build
MPI_BUILD_DIR := build_mpi

CU_OBJS  := $(patsubst src/%.cu, $(BUILD_DIR)/%.o, $(ALL_CU_SRCS))
CPP_OBJS := $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(ALL_CPP_SRCS))
ALL_OBJS := $(CU_OBJS) $(CPP_OBJS)
MPI_CU_OBJS  := $(patsubst src/%.cu, $(MPI_BUILD_DIR)/%.o, $(ALL_CU_SRCS))
MPI_CPP_OBJS := $(patsubst src/%.cpp, $(MPI_BUILD_DIR)/%.o, $(ALL_CPP_SRCS))
MPI_OBJS     := $(MPI_CU_OBJS) $(MPI_CPP_OBJS)

CUDA_MATH_LINK := $(if $(CUBLAS_LIB_DIR),-L$(CUBLAS_LIB_DIR) -Xlinker -rpath=$(CUBLAS_LIB_DIR),) \
                   -lcublas -lcublasLt $(NVTX_LINK)
NCCL_LINK := $(if $(NCCL_LIB_DIR),-L$(NCCL_LIB_DIR) -Xlinker -rpath -Xlinker $(NCCL_LIB_DIR) -lnccl,)

TARGET := $(BUILD_DIR)/train

.PHONY: all clean tests mpi sync_bench alpha_beta_bench figures test_gemm test_attn test_ln

all: $(TARGET)

$(TARGET): $(ALL_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

MPI_TARGET := $(MPI_BUILD_DIR)/train_mpi

mpi: $(MPI_TARGET)

$(MPI_TARGET): $(MPI_OBJS) | $(MPI_BUILD_DIR)
	$(MPICXX) -o $@ $^ $(MPI_LINK_FLAGS) -L$(CUDA_LIB_DIR) -lcudart \
	    -L$(CUBLAS_LIB_DIR) -Wl,-rpath,$(CUBLAS_LIB_DIR) -lcublas -lcublasLt \
	    $(NCCL_LINK) $(NVTX_LINK) $(OPENMP_LINK)

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(MPI_BUILD_DIR)/%.o: src/%.cu | $(MPI_BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_MPI_FLAGS) -c $< -o $@

$(MPI_BUILD_DIR)/%.o: src/%.cpp | $(MPI_BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_MPI_FLAGS) -x cu -c $< -o $@

$(MPI_BUILD_DIR):
	mkdir -p $(MPI_BUILD_DIR)

TEST_GEMM    := $(BUILD_DIR)/test_gemm
TEST_ATTN    := $(BUILD_DIR)/test_attention
TEST_LN      := $(BUILD_DIR)/test_layernorm
TEST_MODEL   := $(BUILD_DIR)/test_model_reference
TEST_FUSION  := $(BUILD_DIR)/test_fusion_benchmark
PROFILE_FUSION := $(BUILD_DIR)/profile_fusion_kernels
PROFILE_HOTSPOTS := $(BUILD_DIR)/profile_training_hotspots

KERNEL_OBJS_ONLY := $(patsubst src/%.cu, $(BUILD_DIR)/%.o, $(KERNEL_SRCS))
MODEL_OBJS_ONLY  := $(KERNEL_OBJS_ONLY) $(BUILD_DIR)/model/transformer.o \
                    $(BUILD_DIR)/model/distributed.o

TEST_BENCH   := $(BUILD_DIR)/benchmark_all
SYNC_BENCH   := $(MPI_BUILD_DIR)/benchmark_gradient_sync
SYNC_BENCH_OBJ := $(MPI_BUILD_DIR)/benchmark_gradient_sync.o
ALPHA_BETA_BENCH := $(MPI_BUILD_DIR)/benchmark_allreduce_alpha_beta
ALPHA_BETA_BENCH_OBJ := $(MPI_BUILD_DIR)/benchmark_allreduce_alpha_beta.o

tests: $(TEST_GEMM) $(TEST_ATTN) $(TEST_LN) $(TEST_MODEL) $(TEST_FUSION) $(TEST_BENCH) $(PROFILE_FUSION) $(PROFILE_HOTSPOTS)

$(TEST_GEMM): tests/test_gemm.cu $(KERNEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

$(TEST_ATTN): tests/test_attention.cu $(KERNEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

$(TEST_LN): tests/test_layernorm.cu $(KERNEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

$(TEST_MODEL): tests/test_model_reference.cu $(MODEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

$(TEST_FUSION): tests/test_fusion_benchmark.cu $(KERNEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

$(PROFILE_FUSION): tests/profile_fusion_kernels.cu $(KERNEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

$(PROFILE_HOTSPOTS): tests/profile_training_hotspots.cu $(KERNEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

$(TEST_BENCH): tests/benchmark_all.cu $(KERNEL_OBJS_ONLY) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_MATH_LINK)

sync_bench: $(SYNC_BENCH)

alpha_beta_bench: $(ALPHA_BETA_BENCH)

figures:
	bash scripts/make_figures.sh

$(SYNC_BENCH_OBJ): tests/benchmark_gradient_sync.cu | $(MPI_BUILD_DIR)
	$(NVCC) $(NVCC_MPI_FLAGS) -c $< -o $@

$(SYNC_BENCH): $(SYNC_BENCH_OBJ) \
               $(MPI_BUILD_DIR)/model/distributed.o \
               $(MPI_BUILD_DIR)/kernels/activations.o | $(MPI_BUILD_DIR)
	$(MPICXX) -o $@ $^ $(MPI_LINK_FLAGS) -L$(CUDA_LIB_DIR) -lcudart \
	    $(NCCL_LINK) $(NVTX_LINK) $(OPENMP_LINK)

$(ALPHA_BETA_BENCH_OBJ): tests/benchmark_allreduce_alpha_beta.cu | $(MPI_BUILD_DIR)
	$(NVCC) $(NVCC_MPI_FLAGS) -c $< -o $@

$(ALPHA_BETA_BENCH): $(ALPHA_BETA_BENCH_OBJ) | $(MPI_BUILD_DIR)
	$(MPICXX) -o $@ $^ $(MPI_LINK_FLAGS) -L$(CUDA_LIB_DIR) -lcudart \
	    $(NCCL_LINK) $(NVTX_LINK) $(OPENMP_LINK)

test_gemm: $(TEST_GEMM)
	srun -p gpu-turing --gres=gpu:1 ./$(TEST_GEMM)

test_attn: $(TEST_ATTN)
	srun -p gpu-turing --gres=gpu:1 ./$(TEST_ATTN)

test_ln: $(TEST_LN)
	srun -p gpu-turing --gres=gpu:1 ./$(TEST_LN)

clean:
	rm -rf $(BUILD_DIR) $(MPI_BUILD_DIR)
