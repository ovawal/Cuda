#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace std;

/**********************************************************
 * error checking (unchanged from skeleton)
 ***********************************************************/
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
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
    // MASSIVELY PARALLEL SORT USING NVIDIA CUB (the real deal)
    /////////////////////////////////////////////////////////////////////

    int *d_arr = nullptr;
    void *d_temp_storage = nullptr;
    size_t temp_bytes = 0;

    CudaSafeCall( cudaMalloc(&d_arr, size * sizeof(int)) );
    CudaSafeCall( cudaMemcpy(d_arr, array, size * sizeof(int), cudaMemcpyHostToDevice) );

    // First call: compute required temporary storage size
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_bytes,
        d_arr, d_arr, size);

    // Allocate temporary storage
    CudaSafeCall( cudaMalloc(&d_temp_storage, temp_bytes) );

    // Launch the actual sort — thousands of threads, ultra fast
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_bytes,
        d_arr, d_arr, size);

    CudaCheckError();

    // Copy sorted result back
    CudaSafeCall( cudaMemcpy(array, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost) );

    // Cleanup
    CudaSafeCall( cudaFree(d_arr) );
    CudaSafeCall( cudaFree(d_temp_storage) );

    /////////////////////////////////////////////////////////////////////

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTotal, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cerr << "Total time in seconds: " << timeTotal / 1000.0 << endl;

    // Print sorted array
    for (int i = 0; i < size; ++i) {
        cout << array[i];
        if (i < size - 1) cout << " ";
    }
    cout << endl;

    delete[] array;
    return 0;
}
