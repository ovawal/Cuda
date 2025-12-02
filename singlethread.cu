#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace std;


 // error checking 

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
        err = cudaThreadSynchronize();
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

// SINGLE THREAD BOTTOM-UP MERGE SORT (NO RECURSION!)

__device__ void merge(int arr[], int temp[], int left, int mid, int right)
{
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right)
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i <= mid)  temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int p = left; p <= right; p++) arr[p] = temp[p];
}

__global__ void matavgKernel(int *data, int *temp, int n)
{
    // Only one thread runs the entire sort
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Bottom-up merge sort (non-recursive)
        for (int width = 1; width < n; width *= 2) {
            for (int i = 0; i < n; i += 2 * width) {
                int left  = i;
                int mid   = min(i + width, n);
                int right = min(i + 2 * width, n);
                if (mid < right)
                    merge(data, temp, left, mid - 1, right - 1);
            }
            // Copy temp back to data after each pass
            for (int i = 0; i < n; i++) data[i] = temp[i];
        }
    }
}

int main( int argc, char* argv[] )
{
    int * array;
    int size, seed;

    // Only 2 arguments
    if( argc < 3 ){
        cerr << "usage: " << argv[0] << " [N] [seed]\n";
        exit( -1 );
    }

    {
        stringstream ss1( argv[1] );
        ss1 >> size;
    }
    {
        stringstream ss1( argv[2] );
        ss1 >> seed;
    }

    array = makeRandArray( size, seed );

    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord( startTotal, 0 );

    /////////////////////////////////////////////////////////////////////
    // YOUR CODE HERE
    /////////////////////////////////////////////////////////////////////

    int *d_data, *d_temp;
    CudaSafeCall( cudaMalloc(&d_data, size * sizeof(int)) );
    CudaSafeCall( cudaMalloc(&d_temp, size * sizeof(int)) );
    CudaSafeCall( cudaMemcpy(d_data, array, size * sizeof(int), cudaMemcpyHostToDevice) );

    // EXACTLY ONE THREAD
    matavgKernel<<<1,1>>>(d_data, d_temp, size);
    CudaCheckError();

    CudaSafeCall( cudaMemcpy(array, d_data, size * sizeof(int), cudaMemcpyDeviceToHost) );
    CudaSafeCall( cudaFree(d_data) );
    CudaSafeCall( cudaFree(d_temp) );

    // Timer
    cudaEventRecord( stopTotal, 0 );
    cudaEventSynchronize( stopTotal );
    cudaEventElapsedTime( &timeTotal, startTotal, stopTotal );
    cudaEventDestroy( startTotal );
    cudaEventDestroy( stopTotal );

    cerr << "Total time in seconds: " << timeTotal / 1000.0 << endl;

    // Always print
    for( int i = 0; i < size; i++ ){
        cout << array[i];
        if( i < size-1 ) cout << " ";
    }
    cout << endl;

    delete[] array;
    return 0;
}
