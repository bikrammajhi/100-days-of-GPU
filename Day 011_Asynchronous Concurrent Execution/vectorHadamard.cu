#include <ctime>                          // for time() to seed the RNG
#include <cstdlib>                        // for rand(), srand(), and std::exit
#include <iostream>                       // for std::cout, std::endl
#include <cuda_runtime.h>                 // for CUDA runtime API
#include <device_launch_parameters.h>     // for blockIdx, threadIdx, etc.

/* 
Hadamard Product of Two Vectors:
--------------------------
Given two vectors 'a' and 'b' of the same dimension 'n', their element-wise product is:
  a ⊙ b = (a1*b1, a2*b2, ..., an*bn)
Common in neural nets and image processing for per-element operations.
*/

__global__ void vectorHadamard(                // CUDA kernel computing c = a ⊙ b
    const float *a, const float *b, float *c,
    int numElements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index

  if (idx < numElements) {                          // bounds check
    c[idx] = a[idx] * b[idx];                       // element-wise multiply
  }
}

int main() {
  srand(static_cast<unsigned int>(time(0)));  // seed host RNG with current time

  int numElements = 100;                      // vector size
  size_t size = numElements * sizeof(float);  // total bytes per vector

  float *a_h, *b_h, *c_h;                     // host pointers
  float *a_d, *b_d, *c_d;                     // device pointers

  a_h = (float *)malloc(size);                // allocate host memory for a
  b_h = (float *)malloc(size);                // allocate host memory for b
  c_h = (float *)malloc(size);                // allocate host memory for c

  for (int idx = 0; idx < numElements; ++idx) {
    a_h[idx] = static_cast<float>(rand()) /   // random float in [0,1]
               static_cast<float>(RAND_MAX);
    b_h[idx] = static_cast<float>(rand()) /   // random float in [0,1]
               static_cast<float>(RAND_MAX);
  }

  cudaMalloc((void **)&a_d, size);            // allocate device memory for a
  cudaMalloc((void **)&b_d, size);            // allocate device memory for b
  cudaMalloc((void **)&c_d, size);            // allocate device memory for c

  cudaMemcpy(a_d, a_h, size,                  // copy a_h → a_d
             cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size,                  // copy b_h → b_d
             cudaMemcpyHostToDevice);

  int ThreadsPerBlock = 256;                  // CUDA threads per block
  int blocksPerGrid =                        // blocks to cover all elements
    (numElements + ThreadsPerBlock - 1) / ThreadsPerBlock;

  vectorHadamard<<<blocksPerGrid, ThreadsPerBlock>>>(  // launch kernel
    a_d, b_d, c_d, numElements);
  cudaDeviceSynchronize();                   // wait for GPU to finish

  cudaMemcpy(c_h, c_d, size,                  // copy c_d → c_h
             cudaMemcpyDeviceToHost);

  for (int idx = 0; idx < numElements; ++idx) {
    std::cout << a_h[idx] << " * " << b_h[idx]   // print results
              << " = " << c_h[idx] << std::endl;
  }

  free(a_h); free(b_h); free(c_h);            // free host memory
  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); // free device memory

  return 0;                                    // end of program
}
