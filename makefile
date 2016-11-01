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
CCFLAGS = -O2 -std=c++11

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) -g -lineinfo
HOST_FLAGS = -Xcompiler -fopenmp,-pthread
DIAGNOSTIC_FLAGS = -res-usage -Xptxas -warn-lmem-usage,-warn-spills
INCLUDES = -I ./include/ -I $(CUDASIEVE_DIR)/include/ -I $(CUDA_DIR)/include/ -I $(UINT128_DIR) -I ./
# U128 = lib/libu128.a
LIBNAME = cudasieve
CC_LIBS = -lm -lstdc++
NVCC_LIBS = -lcudart $(CC_LIBS) -l$(LIBNAME)

CLI_SRC_DIR = src
SRC_DIR = src
OBJ_DIR = obj

_OBJS =\
 main.o \
 P2.o \
 trivial.o\
 V.o\
 sieve/lpf_mu.o \
 S0.o \
 phi.o \
 S3.o \
 li.o \
 mutest.o \
 sieve/phisieve_device.o \
 sieve/phisieve_host.o \
 sieve/S2_hard_host.o \
 sieve/S2_hard_device.o \
 general/tools.o \
 general/device_functions.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))

PHI_SRC = src/utils/phi.cpp

PHI = phi
MAIN = pix
CS_LIB = $(CUDASIEVE_DIR)/lib$(LIBNAME).a

$(MAIN): $(OBJS) $(CS_LIB)
	@ $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(NVCC_LIBS) $(OBJS) -o $@
	@echo "     CUDA     " $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@ $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<
	@echo "     CUDA     " $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@ $(CC) $(CCFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "     CXX      " $@

# answer check function
$(PHI): $(PHI_SRC) $(CS_LIB)
	@ $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(CC_LIBS) -l$(LIBNAME) $< -o $@
	@echo "     CUDA     " $@

# test function
testu128: src/test128.cu
	$(NVCC) $(NVCC_FLAGS) -g $(INCLUDES) -L $(CUDASIEVE_DIR) -I ./ $(CC_LIBS) -l$(LIBNAME) $^ -o $@

ps: src/sieve/phisieve_device.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o src/sieve/phisieve_device.o $<

clean:
	rm -f obj/*.o obj/sieve/*.o obj/general/*.o cstest phi pix testu128 u128 lib/*.a
