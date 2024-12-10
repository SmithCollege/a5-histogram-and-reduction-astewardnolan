#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>


__global__ void reduction_kernel(int *input, int *output, int n) {
    __shared__ int partialSum[1024];  // 1024 is the maximum block size, adjust as needed
    
    // Calculate global thread index
    int t = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread loads ONE element into shared memory
    if (t < n) {
        partialSum[threadIdx.x] = input[t];
    } else {
        partialSum[threadIdx.x] = 0;  // Padding 
    }
    
    __syncthreads();  

    // Perform reduction w a stride that doubles with each step
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0 && threadIdx.x + stride < blockDim.x) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();  
    }

    // Write the result from the first thread of the block to the output array
    if (threadIdx.x == 0) {
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


int main() {
    int n = 1024;  // Array size (1M elements)
    int *h_input = (int*)malloc(n * sizeof(int));
    int *d_input, *d_output;

    // Initialize input array with values (for simplicity, all elements are 1)
    for (int i = 0; i < n; i++) {
        h_input[i] = i+1;  // Assign values 1, 2, 3, ..., n
    }

    double start = get_clock();


    // Allocate device memory
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, (n / 1024) * sizeof(int));  // One value per block

    // Copy data to device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid size
    int blockSize = 1024;  // Max threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the reduction kernel
    reduction_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Allocate memory to store partial sums on the host
    int *partial_output = (int*)malloc(numBlocks * sizeof(int));
    
    // Copy the partial sums from device to host
    cudaMemcpy(partial_output, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform final reduction on the host (sum the partial results)
    int totalSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += partial_output[i];
    }
    double end = get_clock();

    // Output the result
    printf("The sum of the array is: %d\n", totalSum);
    printf("time per call: %f ns\n", (end-start) );


    // Clean up
    free(h_input);
    free(partial_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}




