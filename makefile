# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = sm_61
# Compilers to use
CUDASIEVE_DIR = /home/curtis/CUDASieve
NVCC = $(CUDA_DIR)/bin/nvcc
CC = clang
# Flags for the host compiler passed from nvcc
CCFLAGS = -O2 std=c++11

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /bin/g++-5 -std=c++11 -arch=$(GPU_ARCH)
HOST_FLAGS = -Xcompiler -fopenmp,-pthread
INCLUDES = -I ./include/ -I $(CUDASIEVE_DIR)/include/ -I $(CUDA_DIR)/include/
CC_LIBS = -lm -lstdc++ -lgmp -lgmpxx
NVCC_LIBS = -lcudart $(CC_LIBS)

CLI_SRC_DIR = src
SRC_DIR = src
OBJ_DIR = obj

_OBJS = P2.o trivial.o V.o sieve/lpf_mu.o S0.o phi.o S3.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))

PHI_SRC = src/utils/phi.cpp
SRC = src/P2.cu src/trivial.cu src/V.cu src/sieve/lpf_mu.cu src/S0.cu src/phi.cu src/S3.cu
MAIN_SRC = src/main.cpp

U128 = u128
PHI = phi
MAIN = pix
LIBNAME = cudasieve
CS_LIB = $(CUDASIEVE_DIR)/lib$(LIBNAME).a

$(MAIN): $(OBJS) $(CS_LIB) $(MAIN_SRC)
	$(NVCC) $(NVCC_FLAGS) $(HOST_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(NVCC_LIBS) -l$(LIBNAME) $(OBJS) $(MAIN_SRC) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

$(PHI): $(PHI_SRC) $(CS_LIB)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(CC_LIBS) -l$(LIBNAME) $< -o $@

$(U128): src/test128.cu
	$(NVCC) $(NVCC_FLAGS) -g $(INCLUDES) -L $(CUDASIEVE_DIR) $(CC_LIBS) -l$(LIBNAME) $^ -o $@

clean:
	rm -f obj/*.o cstest phi pix
