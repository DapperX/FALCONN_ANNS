all:
	python3 gen.py
	clang-format --style=Google -i python_wrapper.cc
	g++ -march=native -O3 -shared -std=c++17 -fPIC -I ../../../external/pybind11/include `python3-config --cflags --ldflags` python_wrapper.cc -o _falconn.so -I ../../include -I ../../../external/eigen -I ../../../external/simple-serializer -I ../../../external/parlaylib/include
