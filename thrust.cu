#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <sstream>
#include <cstdlib>


using namespace std;       

define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 )   // do-while(0)
    do {
        if ( cudaSuccess != err ) {
            fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );
#pragma warning( pop )
#endif
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 )
    do {
        cudaError_t err = cudaGetLastError();
        if ( cudaSuccess != err ) {
            fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
        err = cudaThreadSynchronize();
        if( cudaSuccess != err ) {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );
#pragma warning( pop )
#endif
    return;
}

//   random-array generator 
int * makeRandArray( const int size, const int seed )
{
    srand( seed );
    int * array = new int[ size ];
    for( int i = 0; i < size; i ++ ) {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

//   kernel placeholder                             
__global__ void matavgKernel( ) { }


int main( int argc, char* argv[] )
{
    int * array;                // the poitner to the array of rands
    int size, seed;             // values for the size of the array
    bool printSorted = true;   // and the seed for generating
                                // random numbers

    // check the command line args
    if( argc < 3 ){
        cerr << "usage: " << argv[0] << " [amount of random nums to generate] [seed value for random number generation]" << " [1 to print sorted array, 0 otherwise]" << endl;
        exit( -1 );
    }

    // convert cstrings to ints
    {
        stringstream ss1( argv[1] );
        ss1 >> size;
    }
    {
        stringstream ss1( argv[2] );
        ss1 >> seed;
    }

    // get the random numbers
    array = makeRandArray( size, seed );       
    
    // create a cuda timer to time execution
    
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord( startTotal, 0 );

       // end of cuda timer creation

       /////////////////////////////////////////////////////////////////////
    /////////////////////// YOUR CODE HERE ///////////////////////
    /////////////////////////////////////////////////////////////////////

    // Thrust sort on the GPU
    thrust::device_vector<int> d_vec( array, array + size );
    thrust::sort( thrust::device, d_vec.begin(), d_vec.end() );

    // optional printing of the sorted array
    if( printSorted ){
        thrust::host_vector<int> h_vec = d_vec;
        for( int i = 0; i < size; ++i ){
            cout << h_vec[i];
            if( i < size-1 ) cout << " ";
        }
        cout << endl;
    }

    CudaCheckError();

     // Stop and destroy the cuda timer
    
    cudaEventRecord( stopTotal, 0 );
    cudaEventSynchronize( stopTotal );
    cudaEventElapsedTime( &timeTotal, startTotal, stopTotal );
    cudaEventDestroy( startTotal );
    cudaEventDestroy( stopTotal );

    // end of cuda timer destruction
     

    cerr << "Total time in seconds: " << timeTotal / 1000.0 << endl;

    if( printSorted ){
        ///////////////////////////////////////////////
        /// Your code to print the sorted array here
        // ///////////////////////////////////////////////
    }

    delete[] array;
    return 0;
}

