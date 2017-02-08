# KOKKOS setup (OpenMP or CUDA backend)
# here we assume KOKKOS_PATH is setup by "module load kokkos"
ifndef KOKKOS_PATH
$(error You must set env variable KOKKOS_PATH to the directory where is Makefile.kokkos)
endif
include $(KOKKOS_PATH)/Makefile.kokkos

EXE_PREFIX=euler2d

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = $(NVCC_WRAPPER)
CXXFLAGS = -ccbin g++ -I. -Wall -DUSE_DOUBLE -DCUDA -lineinfo
LDFLAGS =
EXE = $(EXE_PREFIX).cuda
else
CXX = g++
CXXFLAGS = -g -O3 -I. -Wall --std=c++11 -DUSE_DOUBLE
#CXX = xlc++
#CXXFLAGS = -O4 -I. -qarch=pwr8 -qtune -qsmp=omp -std=c++11 -DUSE_DOUBLE
EXE = $(EXE_PREFIX).omp
endif

SRCDIR = $(shell pwd)
SRC = \
	config/inih/ini.cpp \
	config/inih/INIReader.cpp \
	config/ConfigMap.cpp \
	HydroParams.cpp \
	HydroRun.cpp \
	Timer.cpp \
	main.cpp

TMP_OBJ = $(SRC:.c=.o)
OBJ     = $(TMP_OBJ:.cpp=.o)

.DEFAULT_GOAL = all
all: $(EXE)
$(EXE) : $(OBJ)
	echo $@
	$(CXX) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $@

clean:
	rm -f $(OBJ) $(EXE_PREFIX).*

cleandata:
	rm -f *.vti

cleanall: clean
	rm -f *.vti

%.o:    %.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $@
