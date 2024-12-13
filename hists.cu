#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define HISTOGRAM_SIZE 1024
#define BLOCK_SIZE 256  // The number of threads per block

//Sources: https://tatourian.blog/2013/10/29/histogram-on-gpu-using-cuda/

// CUDA Kernel for Strided Histogram Calculation
__global__ void stridedHistogramKernel(unsigned char *d_data, unsigned int *d_histogram, int numElements) {
    // Shared memory for partial histograms within each block
    __shared__ unsigned int sharedHist[HISTOGRAM_SIZE];

    // Thread index in the global array
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared memory histogram to 0
    if (threadIdx.x < HISTOGRAM_SIZE) {
        sharedHist[threadIdx.x] = 0;
    }

    __syncthreads();

    // Each thread processes elements in a strided fashion
    for (int i = tid; i < numElements; i += blockDim.x * gridDim.x) {
        atomicAdd(&sharedHist[d_data[i]], 1);  // Increment histogram bin using atomic add
    }

    __syncthreads();

    // After all threads process their elements, write the partial histogram from shared memory to global memory
    if (threadIdx.x < HISTOGRAM_SIZE) {
        atomicAdd(&d_histogram[threadIdx.x], sharedHist[threadIdx.x]);
    }
}

double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { 
        printf("gettimeofday error"); 
        }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
    int numElements = 1000000;  // Total number of elements in the input array
    unsigned char *h_data = new unsigned char[numElements];
    unsigned int *h_histogram = new unsigned int[HISTOGRAM_SIZE]();

    // Initialize input data with random values between 0-255 (for example purposes)
    for (int i = 0; i < numElements; i++) {
        h_data[i] = rand() % HISTOGRAM_SIZE;
    }

    // Device pointers for input data and histogram
    unsigned char *d_data;
    unsigned int *d_histogram;

    // Allocate memory on the device
    cudaMalloc((void**)&d_data, numElements * sizeof(unsigned char));
    cudaMalloc((void**)&d_histogram, HISTOGRAM_SIZE * sizeof(unsigned int));

    // Copy input data from host to device
    cudaMemcpy(d_data, h_data, numElements * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int));  // Initialize histogram to zero
    
    double start = get_clock();



    // Launch the kernel 
    int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;  
    stridedHistogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_data, d_histogram, numElements);

    double end = get_clock();

    printf("time per call: %f ns\n", (end-start) );


    // Copy the result back to host memory
    cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Print the histogram (first 10 bins)
    printf("Histogram:\n");
    for (int i = 0; i < 10; i++) {
        printf("Value %d: %u\n", i, h_histogram[i]);
    }


    // Free memory
    delete[] h_data;
    delete[] h_histogram;
    cudaFree(d_data);
    cudaFree(d_histogram);

    return 0;
}
