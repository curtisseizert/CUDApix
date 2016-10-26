# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = sm_61
# Compilers to use
CUDASIEVE_DIR = /home/curtis/CUDASieve
UINT128_DIR = /home/curtis/CUDA-uint128
NVCC = $(CUDA_DIR)/bin/nvcc
CC = g++
# Flags for the host compiler passed from nvcc
CCFLAGS = -O2 std=c++11

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) -g
HOST_FLAGS = -Xcompiler -fopenmp,-pthread
INCLUDES = -I ./include/ -I $(CUDASIEVE_DIR)/include/ -I $(CUDA_DIR)/include/ -I $(UINT128_DIR)
U128 = lib/libu128.a
LIBNAME = cudasieve
CC_LIBS = -lm -lstdc++
NVCC_LIBS = -lcudart $(CC_LIBS) -lu128 -l$(LIBNAME)

CLI_SRC_DIR = src
SRC_DIR = src
OBJ_DIR = obj

_OBJS = P2.o trivial.o V.o sieve/lpf_mu.o S0.o phi.o S3.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))

PHI_SRC = src/utils/phi.cpp
SRC = src/V.cu src/sieve/lpf_mu.cu src/S0.cu src/phi.cu src/S3.cu
U128_SRC = src/device128/device128.cu
U128_DEF = $(UINT128_DIR)/uint128_t.cu
MAIN_SRC = src/main.cpp

PHI = phi
MAIN = pix
CS_LIB = $(CUDASIEVE_DIR)/lib$(LIBNAME).a

$(MAIN): $(OBJS) $(CS_LIB) $(MAIN_SRC) $(U128)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) -L ./lib $(NVCC_LIBS) $(OBJS) $(MAIN_SRC) -o $@

$(U128): $(U128_SRC) $(U128_DEF)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -lib $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# answer check function
$(PHI): $(PHI_SRC) $(CS_LIB)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(CC_LIBS) -l$(LIBNAME) $< -o $@

# test function
testu128: src/test128.cu
	$(NVCC) $(NVCC_FLAGS) -g $(INCLUDES) -L $(CUDASIEVE_DIR) -I ./ $(CC_LIBS) -l$(LIBNAME) $^ -o $@

clean:
	rm -f obj/*.o cstest phi pix lib/*.a
