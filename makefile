# Makefile for Tour d'Algorithms  Cuda Sorting  programs

# Compiler and flags
NVCC = nvcc
PROGRAMS = thrust singlethread multithread

# Default target: build all programs
all: $(PROGRAMS)

# thrust
thrust: thrust.cu
	$(NVCC)  $< -o $@

# singlethread
singlethread: singlethread.cu
	$(NVCC)   $< -o $@

# multithread
multithread: multithread.cu
	$(NVCC)   $< -o $@


# Generate time for the sortng algorithms using runScript.sh
time: all
	./script.sh

# Generate plots using plots.py
plots: test
	python3 plots.py

# Clean up generated files
clean:
	rm -f $(PROGRAMS) timeData.csv *.png

.PHONY: all test plots clean
