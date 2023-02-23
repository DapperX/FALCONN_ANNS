CXX = g++
#CXX = clang++
DIR_PARLAY = /home/dappur/Software/parlaylib/include
DIR_FALCONN = ../../include
DIR_EIGEN = ../../../external/eigen
DIR_SERIALIZER = -I ../../../external/simple-serializer
DIR_NLOHMANN = -I ../../../external/nlohmann
DIR_INCLUDE = $(DIR_PARLAY) $(DIR_FALCONN) $(DIR_EIGEN) $(DIR_SERIALIZER) $(DIR_NLOHMANN)

CXXFLAGS_BASIC = -std=c++17 -Wall -Wextra -march=native -pthread $(DIR_INCLUDE:%=-I%)
CXXFLAGS_DEBUG = -O0 -g
CXXFLAGS_RELEASE = -O3
CXXFLAGS_HDF5 = -DSUPPORT_HDF5 -lhdf5 -lhdf5_cpp
CXXFLAGS_IGNORE_SYMBOL = -Wl,--unresolved-symbols=ignore-all
CXXFLAGS_JEMALLOC = -L$(shell jemalloc-config --libdir) -ljemalloc

calc_recall: calc_recall.cpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) -o $@
