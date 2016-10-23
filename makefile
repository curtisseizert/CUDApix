# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = sm_61
# Compilers to use
CUDASIEVE_DIR = /home/curtis/CUDASieve
NVCC = $(CUDA_DIR)/bin/nvcc
CC = clang
# Flags for the host compiler
CCFLAGS = -O2 -std=c++11 -g

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) --ptxas-options=-dlcm=cg -O2 -g -Xcompiler -fopenmp,-pthread -lineinfo
NVCC_PROFILE_FLAGS = -lineinfo

INCLUDES = -I ./include/ -I $(CUDASIEVE_DIR)/include/ -I $(CUDA_DIR)/include/
CC_LIBS = -lm -lstdc++ -lprimesieve -lgmp -lgmpxx
NVCC_LIBS = -lcudart $(CC_LIBS)

CLI_SRC_DIR = src
SRC_DIR = src
OBJ_DIR = obj

PHI_SRC = src/utils/phi.cpp
SRC = src/main.cu src/P2.cu src/trivial.cu src/V.cu src/sieve/lpf_mu.cu src/S0.cu

PHI = phi
MAIN = cstest
LIBNAME = cudasieve
CS_LIB = $(CUDASIEVE_DIR)/lib$(LIBNAME).a

$(MAIN): $(SRC) $(CS_LIB)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(NVCC_LIBS) -l$(LIBNAME) $^ -o $@

$(PHI): $(PHI_SRC) $(CS_LIB)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(CC_LIBS) -l$(LIBNAME) $^ -o $@

clean:
	rm -f obj/*.o cstest phi
