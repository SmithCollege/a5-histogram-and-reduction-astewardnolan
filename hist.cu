#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>


#define HISTOGRAM_SIZE 1024
#define BLOCK_SIZE 256 

__global__ void histogramKernel(unsigned char *d_data, unsigned int *d_histogram, int numElements) {
    // Shared memory for partial histogram
    __shared__ unsigned int sharedHist[HISTOGRAM_SIZE];

    // Thread index
    int t = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize shared histogram to zero
    if (threadIdx.x < HISTOGRAM_SIZE) {
        sharedHist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Each thread processes one element
    if (t < numElements) {
        atomicAdd(&sharedHist[d_data[t]], 1);  // Update the histogram in shared memory atomically
    }

    __syncthreads();

    // Now accumulate the results into global memory (from shared memory)
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
    int numElements = 1000000;  // Total number of elements
    unsigned char *h_data = new unsigned char[numElements];
    unsigned int *h_histogram = new unsigned int[HISTOGRAM_SIZE]();

    // Fill input data with random values between 0-255
    for (int i = 0; i < numElements; i++) {
        h_data[i] = rand() % HISTOGRAM_SIZE;
    }

    // Device pointers
    unsigned char *d_data;
    unsigned int *d_histogram;

    // Allocate device memory
    cudaMalloc((void**)&d_data, numElements * sizeof(unsigned char));
    cudaMalloc((void**)&d_histogram, HISTOGRAM_SIZE * sizeof(unsigned int));

    // Copy input data to device
    cudaMemcpy(d_data, h_data, numElements * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int));  // Initialize histogram to 0

    // Create CUDA events for timing
    double start = get_clock();


    int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_data, d_histogram, numElements);

    double end = get_clock();
    printf("time per call: %f ns\n", (end-start) );



    // Copy histogram result back to host
    cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Print the histogram (for the first 10 values as example)
    printf("Histogram:\n");
    for (int i = 0; i < 10; i++) {
        printf("Value %d: %u\n", i, h_histogram[i]);
    }

    // Print the kernel execution time

    // Clean up
    delete[] h_data;
    delete[] h_histogram;
    cudaFree(d_data);
    cudaFree(d_histogram);

    return 0;
}
