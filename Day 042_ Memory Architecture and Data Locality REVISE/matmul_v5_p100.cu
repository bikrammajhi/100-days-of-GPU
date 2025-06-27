/*
 * GEMM Kernel with cuBLAS Benchmarking
 * Tests multiple matrix sizes: 512, 1024, 2048, 4096, 8192
 * 
 * GEMM Kernel Assumptions:
 * 1. M % BM_dim == 0, N % BN_dim == 0, K % BK_dim == 0
 * 2. NUM_THREADS % BN_dim == 0
 * 3. NUM_THREADS % BK_dim == 0
 * 4. (BM_dim*BK_dim + BK_dim*BN_dim)*sizeof(float) <= shared memory limit
 * 5. (BN_dim / TN_dim) * (BM_dim / TM_dim) == NUM_THREADS
 * 6. BK_dim % 32 == 0, "BK must be multiple of 32 for coalescing, BN_dim % 32 == 0, "BN must be multiple of 32 for coalescing
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << err << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)


// Loading from Global to Shared Memory
__device__ __forceinline__ void tileMemcpy(
    float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int src_stride,
    const unsigned int tile_rows,
    const unsigned int tile_cols,
    const unsigned int NUM_THREADS
) {
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int row_step = NUM_THREADS / tile_cols;
    const unsigned int thread_col = thread_idx % tile_cols;
    const unsigned int thread_row = thread_idx / tile_cols;
    
    #pragma unroll(2)
    for (unsigned int r = thread_row; r < tile_rows; r += row_step) {
        dst[r * tile_cols + thread_col] = src[r * src_stride + thread_col];
    }
}

// Loading from Shared Memory to Registers 
template <int TM_dim, int TN_dim, int BK_dim, int BN_dim>
__device__ __forceinline__ void load_matrix_tiles_to_registers(
    const float* __restrict__ A_block_smem,
    const float* __restrict__ B_block_smem,
    float* __restrict__ A_register,  // Use pointer (saves 1 register)
    float* __restrict__ B_register,
    const int thread_row,
    const int thread_col,
    const int k_inner
) {
    // Load A fragment (column-major)
    #pragma unroll
    for (int tm = 0; tm < TM_dim; tm++) {
        A_register[tm] = A_block_smem[(thread_row + tm) * BK_dim + k_inner];
    }
    
    // Load B fragment (row-major)
    #pragma unroll
    for (int tn = 0; tn < TN_dim; tn++) {
        B_register[tn] = B_block_smem[k_inner * BN_dim + thread_col + tn];
    }
}

template <
    unsigned int BM_dim,
    unsigned int BN_dim,
    unsigned int BK_dim,
    unsigned int TM_dim,
    unsigned int TN_dim,
    unsigned int NUM_THREADS
>
__global__ void sgemm_v2_kernel(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {
    // Validate thread configuration
    static_assert((BN_dim / TN_dim) * (BM_dim / TM_dim) == NUM_THREADS, 
                 "Invalid thread count");
    // For Coalescing
    static_assert(BK_dim % 32 == 0, "BK must be multiple of 32 for coalescing");
    static_assert(BN_dim % 32 == 0, "BN must be multiple of 32 for coalescing");
                 
    // 1. Leading dimensions
    const unsigned int A_stride = K;
    const unsigned int B_stride = N;
    const unsigned int C_stride = N;

    // 2. Block index
    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;

    // 3. Thread index
    const unsigned int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int threads_per_block_row = BN_dim / TN_dim;                                // One thread is responsible for calculating TM_dim*TN_dim element in the block
    const unsigned int thread_row = (thread_id / threads_per_block_row) * TM_dim;
    const unsigned int thread_col = (thread_id % threads_per_block_row) * TN_dim;

    // 4. Shared memory allocation
    extern __shared__ float shmem[];
    float* A_block_smem = shmem;
    float* B_block_smem = &shmem[BM_dim * BK_dim];

    // 5. Main computation
    const unsigned int num_block_tiles_k = K / BK_dim;
    float acc_register[TM_dim][TN_dim] = {0.};                                                 // Register to accumulate TM_dim*TN_dim element for each thread

    float A_register[TM_dim] = {0.};                               
    float B_register[TN_dim] = {0.};   

    #pragma unroll(4)
    for (unsigned int block_k = 0; block_k < num_block_tiles_k; ++block_k) {
        // Load tiles
        float* A_block_gmem = &A[block_m * BM_dim * A_stride + block_k * BK_dim];
        float* B_block_gmem = &B[block_k * BK_dim * B_stride + block_n * BN_dim];

        tileMemcpy(A_block_gmem, A_block_smem, A_stride, BM_dim, BK_dim, NUM_THREADS);
        tileMemcpy(B_block_gmem, B_block_smem, B_stride, BK_dim, BN_dim, NUM_THREADS);
        __syncthreads();
        
        #pragma unroll
        for (unsigned int k_inner = 0; k_inner < BK_dim; ++k_inner) { 
            // Load both fragments 
            load_matrix_tiles_to_registers<TM_dim, TN_dim, BK_dim, BN_dim>(A_block_smem, B_block_smem, A_register, B_register, thread_row, thread_col, k_inner);

            #pragma unroll
            for(unsigned int tm = 0; tm < TM_dim; ++tm){

                #pragma unroll
                for(unsigned int tn = 0; tn < TN_dim; ++tn){
                acc_register[tm][tn] += A_register[tm] * B_register[tn];
                }
            }
            
        }
        __syncthreads();
    }

    // 6. Store results
    float* C_tile = &C[block_m * BM_dim * C_stride + block_n * BN_dim];
    for (unsigned int tm = 0; tm < TM_dim; ++tm) {
        const unsigned int row_idx = thread_row + tm;
        for(unsigned int tn = 0; tn < TN_dim; ++tn) {
            const unsigned int col_idx = thread_col + tn;
            const unsigned int c_index = row_idx * C_stride + col_idx;
            C_tile[c_index] = alpha * acc_register[tm][tn] + beta * C_tile[c_index];
        }
    }

}

// CPU reference GEMM implementation
void cpu_gemm(
    float* A, float* B, float* C,
    float alpha, float beta,
    unsigned int M, unsigned int N, unsigned int K
) {
    for (unsigned int m = 0; m < M; ++m) {
        for (unsigned int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Compare two matrices
bool verify_results(float* gpu, float* cpu, unsigned int size, float epsilon = 1e-3) {
    unsigned int mismatches = 0;
    float max_error = 0.0f;
    
    for (unsigned int i = 0; i < size; ++i) {
        float error = fabs(gpu[i] - cpu[i]);
        max_error = fmax(max_error, error);
        
        if (error > epsilon) {
            mismatches++;
            if (mismatches <= 5) { // Print first 5 mismatches
                std::cerr << "Mismatch at index " << i 
                          << ": GPU=" << gpu[i] << ", REF=" << cpu[i] 
                          << ", Error=" << error << std::endl;
            }
        }
    }
    
    std::cout << "Max error: " << max_error << ", Mismatches: " << mismatches 
              << "/" << size << std::endl;
    
    return mismatches == 0;
}

// Calculate GFLOPS
double calculate_gflops(unsigned int M, unsigned int N, unsigned int K, float time_ms) {
    double flops = 2.0 * M * N * K; // 2 operations per element (multiply + add)
    return (flops / 1e9) / (time_ms / 1000.0);
}

// Benchmark function
void benchmark_gemm(unsigned int M, unsigned int N, unsigned int K) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Benchmarking GEMM: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Tile dimensions - adjust based on matrix size
    constexpr unsigned int BM_dim = 32*2;
    constexpr unsigned int BN_dim = 32*2;
    constexpr unsigned int BK_dim = 32;
    constexpr unsigned int TM_dim = 4;
    constexpr unsigned int TN_dim = 4;
    constexpr unsigned int NUM_THREADS = BM_dim * BN_dim / (TM_dim * TN_dim);

    // Check if dimensions are compatible
    if (M % BM_dim != 0 || N % BN_dim != 0 || K % BK_dim != 0) {
        std::cout << "Skipping: Matrix dimensions not compatible with tile sizes" << std::endl;
        return;
    }

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_custom(M * N);
    std::vector<float> h_C_cublas(M * N);
    std::vector<float> h_C_original(M * N);

    // Initialize matrices
    srand(42); // Fixed seed for reproducible results
    init_matrix(h_A.data(), M * K);
    init_matrix(h_B.data(), K * N);
    init_matrix(h_C_original.data(), M * N);
    
    // Copy initial C matrix
    std::copy(h_C_original.begin(), h_C_original.end(), h_C_custom.begin());
    std::copy(h_C_original.begin(), h_C_original.end(), h_C_cublas.begin());

    // Allocate device memory
    float *d_A, *d_B, *d_C_custom, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_custom, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, M * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_custom, h_C_custom.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_cublas, h_C_cublas.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.5f;
    const int num_runs = 10;

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up GPU
    for (int i = 0; i < 3; ++i) {
        dim3 grid(N / BN_dim, M / BM_dim);
        dim3 block(BN_dim / TN_dim, BM_dim / TM_dim);
        size_t shmem_size = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(float);
        
        sgemm_v2_kernel<BM_dim, BN_dim, BK_dim, TM_dim, TN_dim, NUM_THREADS>
            <<<grid, block, shmem_size>>>(d_A, d_B, d_C_custom, alpha, beta, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark custom kernel
    float custom_total_time = 0.0f;
    for (int run = 0; run < num_runs; ++run) {
        // Reset C matrix
        CUDA_CHECK(cudaMemcpy(d_C_custom, h_C_original.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 grid(N / BN_dim, M / BM_dim);
        dim3 block(BN_dim / TN_dim, BM_dim / TM_dim);
        size_t shmem_size = (BM_dim * BK_dim + BK_dim * BN_dim) * sizeof(float);
        
        CUDA_CHECK(cudaEventRecord(start));
        sgemm_v2_kernel<BM_dim, BN_dim, BK_dim, TM_dim, TN_dim, NUM_THREADS>
            <<<grid, block, shmem_size>>>(d_A, d_B, d_C_custom, alpha, beta, M, N, K);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        custom_total_time += milliseconds;
    }
    
    float custom_avg_time = custom_total_time / num_runs;
    double custom_gflops = calculate_gflops(M, N, K, custom_avg_time);

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Warm up cuBLAS
    for (int i = 0; i < 3; ++i) {
        CUBLAS_CHECK(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C_cublas, N
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark cuBLAS
    float cublas_total_time = 0.0f;
    for (int run = 0; run < num_runs; ++run) {
        // Reset C matrix
        CUDA_CHECK(cudaMemcpy(d_C_cublas, h_C_original.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C_cublas, N
        ));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        cublas_total_time += milliseconds;
    }
    
    float cublas_avg_time = cublas_total_time / num_runs;
    double cublas_gflops = calculate_gflops(M, N, K, cublas_avg_time);

    // Copy results back for verification
    CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "  Custom Kernel: " << std::setw(8) << custom_avg_time << " ms, " 
              << std::setw(8) << custom_gflops << " GFLOPS" << std::endl;
    std::cout << "  cuBLAS:        " << std::setw(8) << cublas_avg_time << " ms, " 
              << std::setw(8) << cublas_gflops << " GFLOPS" << std::endl;
    std::cout << "  Performance:   " << std::setw(8) << (custom_gflops / cublas_gflops * 100.0) << "% of cuBLAS performance" << std::endl;

    // Verify results
    std::cout << "\nVerification (Custom vs cuBLAS): ";
    if (verify_results(h_C_custom.data(), h_C_cublas.data(), M * N, 1e-3)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_custom));
    CUDA_CHECK(cudaFree(d_C_cublas));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

int main() {
    // Print device information
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;

    // Test different matrix sizes
    std::vector<unsigned int> sizes = {512, 1024, 2048, 4096, 8192};
    
    for (unsigned int size : sizes) {
        try {
            benchmark_gemm(size, size, size);
        } catch (const std::exception& e) {
            std::cerr << "Error benchmarking size " << size << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}