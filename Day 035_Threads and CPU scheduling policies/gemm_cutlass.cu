#include "cutlass/gemm/device/gemm.h"
#include <cuda_runtime.h>
#include <iostream>

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<
    float, ColumnMajor,
    float, ColumnMajor,
    float, ColumnMajor>;

__global__ void fill(float *ptr, int size, float val) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) ptr[idx] = val;
}

int main() {
  int M = 128, N = 128, K = 128;
  float alpha = 1.0f, beta = 0.0f;

  float *A, *B, *C;
  cudaMalloc(&A, M * K * sizeof(float));
  cudaMalloc(&B, K * N * sizeof(float));
  cudaMalloc(&C, M * N * sizeof(float));

  fill<<<(M*K+255)/256, 256>>>(A, M*K, 1.0f);
  fill<<<(K*N+255)/256, 256>>>(B, K*N, 1.0f);
  fill<<<(M*N+255)/256, 256>>>(C, M*N, 0.0f);

  CutlassGemm gemm_op;
  CutlassGemm::Arguments args(
    {M, N, K},
    {A, M},
    {B, K},
    {C, M},
    {C, M},
    {alpha, beta}
  );

  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM failed!" << std::endl;
    return -1;
  }

  std::cout << "CUTLASS GEMM succeeded!" << std::endl;
  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}