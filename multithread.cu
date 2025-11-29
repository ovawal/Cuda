#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <climits>

using namespace std;

/**********************************************************
 * error checking (unchanged)
 ***********************************************************/
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    do { if (cudaSuccess != err) { fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err)); exit(-1); } } while(0);
#endif
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    do {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) { fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err)); exit(-1); }
        err = cudaDeviceSynchronize();
        if (cudaSuccess != err) { fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err)); exit(-1); }
    } while(0);
#endif
    return;
}

int * makeRandArray( const int size, const int seed )
{
    srand( seed );
    int * array = new int[ size ];
    for( int i = 0; i < size; i++ ) {
        array[i] = rand() % 1000000;
    }
    return array;
}

//*******************************//
// MASSIVELY PARALLEL ODD-EVEN MERGE SORT (works for any N)
//*******************************//
__global__ void oddEvenMergeSort(int *arr, int n)
{
    extern __shared__ int s[];

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data (pad with INT_MAX if out of bounds)
    s[tid] = (idx < n) ? arr[idx] : INT_MAX;
    __syncthreads();

    // Odd-Even Transposition Sort inside each block
    for (int phase = 0; phase < n; ++phase) {
        if (phase & 1) {
            // odd phase
            if ((tid & 1) && tid + 1 < blockDim.x)
                if (s[tid] > s[tid + 1]) { int tmp = s[tid]; s[tid] = s[tid+1]; s[tid+1] = tmp; }
        } else {
            // even phase
            if ((tid & 1) == 0 && tid + 1 < blockDim.x)
                if (s[tid] > s[tid + 1]) { int tmp = s[tid]; s[tid] = s[tid+1]; s[tid+1] = tmp; }
        }
        __syncthreads();
    }

    // Write back only valid elements
    if (idx < n)
        arr[idx] = s[tid];
}

int main( int argc, char* argv[] )
{
    int size, seed;
    int *array;

    if (argc < 3) {
        cerr << "usage: " << argv[0] << " [N] [seed]\n";
        return -1;
    }

    stringstream(argv[1]) >> size;
    stringstream(argv[2]) >> seed;

    array = makeRandArray(size, seed);

    cudaEvent_t start, stop;
    float timeTotal = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /////////////////////////////////////////////////////////////////////
    // YOUR CODE HERE – MASSIVELY PARALLEL SORT
    /////////////////////////////////////////////////////////////////////

    int *d_arr;
    CudaSafeCall( cudaMalloc(&d_arr, size * sizeof(int)) );
    CudaSafeCall( cudaMemcpy(d_arr, array, size * sizeof(int), cudaMemcpyHostToDevice) );

    const int THREADS_PER_BLOCK = 1024;
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    oddEvenMergeSort<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_arr, size);
    CudaCheckError();

    CudaSafeCall( cudaMemcpy(array, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost) );
    CudaSafeCall( cudaFree(d_arr) );

    /////////////////////////////////////////////////////////////////////

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTotal, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cerr << "Total time in seconds: " << timeTotal / 1000.0 << endl;

    // Always print the sorted array
    for (int i = 0; i < size; ++i) {
        cout << array[i];
        if (i < size-1) cout << " ";
    }
    cout << endl;

    delete[] array;
    return 0;
}
