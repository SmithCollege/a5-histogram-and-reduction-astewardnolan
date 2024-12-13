#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define HISTOGRAM_SIZE 1024

void generateHistogram(unsigned char *arr, int size, unsigned int *histogram) {
    // Initialize the histogram with 0
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        histogram[i] = 0;
    }

    // Populate the histogram
    for (int i = 0; i < size; i++) {
        if (arr[i] >= 0 && arr[i] < HISTOGRAM_SIZE) {
            histogram[arr[i]]++;
        }
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

void printHistogram(int *histogram) {
    printf("Histogram of values (0-255):\n");
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        printf("%3d: ", i);
        for (int j = 0; j < histogram[i]; j++) {
            printf("*");
        }
        printf("\n");
    }
}

int main() {
    int numElements = 1000000;  // Total number of elements in the input array

    // Dynamically allocate memory for the data and histogram
    unsigned char *h_data = (unsigned char *)malloc(numElements * sizeof(unsigned char));
    unsigned int *h_histogram = (unsigned int *)calloc(HISTOGRAM_SIZE, sizeof(unsigned int));

    if (h_data == NULL || h_histogram == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    // Initialize input data with random values between 0-127 (for histogram)
    for (int i = 0; i < numElements; i++) {
        h_data[i] = rand() % HISTOGRAM_SIZE;
    }


    double start = get_clock();

    // Generate the histogram from the array
    generateHistogram(h_data, numElements, h_histogram);
    double end = get_clock();

    printf("time per call: %f ns\n", (end-start) );

    // Print the histogram
    //printHistogram(histogram);

    return 0;

}
