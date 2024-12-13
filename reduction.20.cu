#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>

#define BLOCK_SIZE 1024

//resources

// Kernel to perform reduction using shared memory
__global__ void reductionKernel(float *input, float *output, int n) {
    // Define shared memory for partial sums (2048 entries for 1024 threads)
    __shared__ float partialSum[2 * BLOCK_SIZE]; 

    unsigned int t = threadIdx.x;  // Thread index within the block
    unsigned int start = 2 * blockIdx.x * blockDim.x;  // Start index in global memory for this block

    // Load the input data into shared memory (two elements per thread)
    if (start + t < n) {
        partialSum[t] = input[start + t];
    } else {
        partialSum[t] = 0.0f;  // Handle boundary conditions (out-of-bounds)
    }

    if (start + blockDim.x + t < n) {
        partialSum[blockDim.x + t] = input[start + blockDim.x + t];
    } else {
        partialSum[blockDim.x + t] = 0.0f;  // Handle boundary conditions (out-of-bounds)
    }

    // Perform reduction
    for (unsigned int stride = blockDim.x; stride >= 1; stride >>= 1) {
        __syncthreads(); 

        // Only the first 'stride' threads are active in this phase
        if (t < stride) {
            partialSum[t] += partialSum[t + stride]; 
        }
    }

    if (t == 0) {
        output[blockIdx.x] = partialSum[0];
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



// Host code to invoke the kernel and manage memory
int main() {
    int n = 2048;  // Example array size (should be a multiple of 2 * BLOCK_SIZE)
    int numBlocks = (n + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);  // Calculate number of blocks

    // Allocate host memory
    float *h_input = new float[n];
    float *h_output = new float[numBlocks];



    // Initialize input array with some values (e.g., sequential numbers)
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1.0f;  // Simple test case where each element is 1.0f
    }
    double start = get_clock();


    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid(numBlocks);
    reductionKernel<<<grid, block>>>(d_input, d_output, n);

    // Copy result from device to host
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction (sum all block results)
    float finalResult = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        finalResult += h_output[i];
    }
    double end = get_clock();

    // Output the result
    printf("time per call: %f ns\n", (end-start) );
    // Print the result
    std::cout << "Final reduction result: " << finalResult << std::endl;

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
