#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>


// reduction tree
int reduction_tree_sum(int* arr, int left, int right) {
    if (left == right) {
        return arr[left];
    }
    // Find the midpoint of the range
    int mid = (left + right) / 2;

    // Recursively compute the sum of the left and right halves
    int left_sum = reduction_tree_sum(arr, left, mid);
    int right_sum = reduction_tree_sum(arr, mid + 1, right);

    // Combine the results from the two halves
    return left_sum + right_sum;
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
  int n = 1024;

  int* arr = (int*)malloc(n * sizeof(int));

  

// Initialize the array with some values
  for (int i = 0; i < n; i++) {
        arr[i] = i + 1;  // Assign values 1, 2, 3, ..., n
    }
  double start = get_clock();


  int result = reduction_tree_sum(arr, 0, n - 1);

  
  double end = get_clock();
  printf("time per call: %f ns\n", (end-start) );
  printf("The sum of the array is: %d\n", result);


  return 0;
}

