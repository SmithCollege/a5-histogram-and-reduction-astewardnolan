#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define HISTOGRAM_SIZE 256

void generateHistogram(int *arr, int size, int *histogram) {
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
    int arr[] = {255, 0, 0, 1, 2, 2, 3, 5, 255, 255, 128, 128, 128, 64, 32, 32, 0, 100};
    int size = sizeof(arr) / sizeof(arr[0]);

    // Create an array to store histogram counts
    int histogram[HISTOGRAM_SIZE];

    double start = get_clock();

    // Generate the histogram from the array
    generateHistogram(arr, size, histogram);
    double end = get_clock();

    printf("time per call: %f ns\n", (end-start) );

    // Print the histogram
    //printHistogram(histogram);

    return 0;
}
