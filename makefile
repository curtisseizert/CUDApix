# Location of the CUDA toolkit
CUDA_DIR = /opt/cuda
# Compute capability of the target GPU
GPU_ARCH = sm_61
# Compilers to use
CUDASIEVE_DIR = /home/curtis/Projects/CUDASieve
UINT128_DIR = /home/curtis/Projects/CUDA-uint128
NVCC = $(CUDA_DIR)/bin/nvcc
CC = g++
# Flags for the host compiler passed from nvcc
CCFLAGS = -O0 -std=c++11 -g -fopenmp

# Flags for nvcc
# ptxas-options=-dlcm=cg (vs. default of ca) is about a 2% performance gain
NVCC_FLAGS = -ccbin /usr/bin/g++-5 -std=c++11 -arch=$(GPU_ARCH) -g -lineinfo -Xcompiler -fopenmp -lineinfo
HOST_FLAGS = -Xcompiler -fopenmp,-pthread
DIAGNOSTIC_FLAGS = -res-usage -Xptxas -warn-lmem-usage,-warn-spills -g -lineinfo
INCLUDES = -I ./include/ -I $(CUDASIEVE_DIR)/include/ -I $(CUDA_DIR)/include/ -I $(UINT128_DIR) -I ./
LIBNAME = cudasieve
CC_LIBS = -lm -lstdc++ -lboost_iostreams -lboost_system -lboost_filesystem
NVCC_LIBS = -lcudart -lm -lstdc++ -l$(LIBNAME) -lboost_iostreams -lboost_system -lboost_filesystem
CLI_SRC_DIR = src
SRC_DIR = src
OBJ_DIR = obj

_OBJS =\
 main.o \
 P2.o \
 trivial.o\
 V.o\
 sieve/lpf_mu.o \
 ordinary.o \
 phi.o \
 S3.o \
 li.o \
 mutest.o \
 pitable.o \
 sieve/phisieve_device.o \
 sieve/phisieve_host.o \
 sieve/S2_hard_host.o \
 sieve/S2_hard_device.o \
 general/leafcount.o \
 general/device_functions.o \
 Gourdon/gourdonvariant.o \
 Gourdon/sigma.o \
 Gourdon/sigma_cpu.o \
 Gourdon/phi_0.o \
 Gourdon/A1.o \
 Gourdon/A2.o \
 Gourdon/A_cpu.o \
 Gourdon/A2_cpu.o \
 Gourdon/B.o \
 Gourdon/C.o \
 Gourdon/C_cpu.o \
 Deleglise-Rivat/deleglise-rivat.o \
 Deleglise-Rivat/A4.o \
 Deleglise-Rivat/A4_cpu.o \
 Deleglise-Rivat/S2.o \
 Deleglise-Rivat/S3.o \
 Deleglise-Rivat/omega12_host.o \
 Deleglise-Rivat/omega12_device.o \
 Deleglise-Rivat/omega3.o \
 Deleglise-Rivat/sigma.o \
 gnuplot/gnuplot.o
OBJS = $(patsubst %,$(OBJ_DIR)/%,$(_OBJS))
DIRS = obj/Deleglise-Rivat \
 obj/general \
 obj/Gourdon \
 obj/sieve \
 obj/gnuplot

PHI_SRC = src/utils/phi.cpp

PHI = phi
MAIN = pix
CS_LIB = $(CUDASIEVE_DIR)/lib$(LIBNAME).a

$(MAIN): $(OBJS) $(CS_LIB)
	@ $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(NVCC_LIBS) $(OBJS) -o $@
	@echo "     CUDA     " $@

$(CS_LIB):
	@+ ./build_cudasieve.sh $(CUDASIEVE_DIR) libcudasieve.a

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<
	@echo "     CUDA     " $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@ $(CC) $(CCFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "     CXX      " $@

# answer check function
$(PHI): $(PHI_SRC) $(CS_LIB)
	@ $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -L $(CUDASIEVE_DIR) $(CC_LIBS) -l$(LIBNAME) $< -o $@
	@echo "     CUDA     " $@

dirs:
	@mkdir obj
	@mkdir $(DIRS)

# test function
testu128: src/test128.cu
	$(NVCC) $(NVCC_FLAGS) -g $(INCLUDES) -L $(CUDASIEVE_DIR) -I ./ $(CC_LIBS) -l$(LIBNAME) $^ -o $@

ps: src/sieve/phisieve_device.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o src/sieve/phisieve_device.o $<

clean:
	rm -f obj/*.o obj/sieve/*.o obj/general/*.o obj/Gourdon/*.o phi pix testu128 u128 lib/*.a
