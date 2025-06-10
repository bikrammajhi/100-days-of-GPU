#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>

#include <cmath>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Error checking macro
#define checkCudaError(call, msg) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// CUDA Event Timer Class
class CudaEventTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    CudaEventTimer() {
        checkCudaError(cudaEventCreate(&start_event), "Failed to create start event");
        checkCudaError(cudaEventCreate(&stop_event), "Failed to create stop event");
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        checkCudaError(cudaEventRecord(start_event), "Failed to record start event");
    }

    void stop() {
        checkCudaError(cudaEventRecord(stop_event), "Failed to record stop event");
    }

    float getElapsedTime() {
        checkCudaError(cudaEventSynchronize(stop_event), "Failed to synchronize stop event");
        float elapsed_time;
        checkCudaError(cudaEventElapsedTime(&elapsed_time, start_event, stop_event),
                      "Failed to get elapsed time");
        return elapsed_time;
    }
};

// Benchmark function using CUDA events
template<typename KernelFunc>
double benchmarkKernelWithEvents(KernelFunc kernel, int warmup_runs = 3, int bench_runs = 10) {
    CudaEventTimer timer;

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        kernel();
    }
    checkCudaError(cudaDeviceSynchronize(), "Warmup failed");

    // Benchmark runs
    std::vector<float> times(bench_runs);
    for (int i = 0; i < bench_runs; ++i) {
        timer.start();
        kernel();
        timer.stop();
        times[i] = timer.getElapsedTime();
    }

    // Calculate average time
    double total_time = 0.0;
    for (float time : times) {
        total_time += time;
    }

    return total_time / bench_runs;
}

// Naive Matrix Multiplication Kernel
__global__ void gemm_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Shared Memory Tiled Matrix Multiplication Kernel
template<int TILE_WIDTH>
__global__ void gemm_tiled(float* A, float* B, float* C, int N) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;
    int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < num_tiles; ++m) {
        if (row < N && m * TILE_WIDTH + tx < N) {
            shared_A[ty][tx] = A[row * N + m * TILE_WIDTH + tx];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        if (m * TILE_WIDTH + ty < N && col < N) {
            shared_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        } else {
            shared_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized Pipelined Matrix Multiplication Kernel
template<int TILE_WIDTH, int NUM_STAGES>
__global__ void gemm_pipelined(float* A, float* B, float* C, int N) {
    // Use cooperative groups for block-scope pipeline
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES> shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    __shared__ float shared_A[NUM_STAGES][TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[NUM_STAGES][TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    // Phase 1: Fill pipeline with initial tiles
    for (int s = 0; s < NUM_STAGES && s < num_tiles; ++s) {
        pipeline.producer_acquire();

        int m = s;
        if (row < N && m * TILE_WIDTH + tx < N) {
            cuda::memcpy_async(block, &shared_A[s][ty][tx], &A[row * N + m * TILE_WIDTH + tx],
                              sizeof(float), pipeline);
        } else {
            shared_A[s][ty][tx] = 0.0f;
        }
        if (m * TILE_WIDTH + ty < N && col < N) {
            cuda::memcpy_async(block, &shared_B[s][ty][tx], &B[(m * TILE_WIDTH + ty) * N + col],
                              sizeof(float), pipeline);
        } else {
            shared_B[s][ty][tx] = 0.0f;
        }

        pipeline.producer_commit();
        block.sync(); // Ensure all threads complete loading
    }

    // Phase 2: Steady-state processing
    for (int m = 0; m < num_tiles; ++m) {
        int stage = m % NUM_STAGES;

        pipeline.consumer_wait();
        block.sync();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_A[stage][ty][k] * shared_B[stage][k][tx];
        }

        block.sync();

        if (m + NUM_STAGES < num_tiles) {
            int next_m = m + NUM_STAGES;
            int next_stage = next_m % NUM_STAGES;

            pipeline.producer_acquire();
            if (row < N && next_m * TILE_WIDTH + tx < N) {
                cuda::memcpy_async(block, &shared_A[next_stage][ty][tx], &A[row * N + next_m * TILE_WIDTH + tx],
                                  sizeof(float), pipeline);
            } else {
                shared_A[next_stage][ty][tx] = 0.0f;
            }
            if (next_m * TILE_WIDTH + ty < N && col < N) {
                cuda::memcpy_async(block, &shared_B[next_stage][ty][tx], &B[(next_m * TILE_WIDTH + ty) * N + col],
                                  sizeof(float), pipeline);
            } else {
                shared_B[next_stage][ty][tx] = 0.0f;
            }

            pipeline.producer_commit();
            block.sync();
        }
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Utility Functions
void initializeMatrix(std::vector<float>& matrix, int N, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dis(gen);
    }
}

bool verifyResults(const std::vector<float>& C1, const std::vector<float>& C2,
                  int N, float tolerance = 1e-3f) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(C1[i] - C2[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": "
                      << C1[i] << " vs " << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void launch_naive_kernel(float* d_A, float* d_B, float* d_C, int N,
                         dim3 grid, dim3 block) {
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);
}

void launch_tiled_kernel(float* d_A, float* d_B, float* d_C, int N,
                         dim3 grid, dim3 block, int tile_width) {
    if (tile_width == 16) {
        gemm_tiled<16><<<grid, block>>>(d_A, d_B, d_C, N);
    } else if (tile_width == 32) {
        gemm_tiled<32><<<grid, block>>>(d_A, d_B, d_C, N);
    }
}

void launch_pipelined_kernel(float* d_A, float* d_B, float* d_C, int N,
                             dim3 grid, dim3 block, int tile_width, int num_stages) {
    if (tile_width == 16 && num_stages == 2) {
        gemm_pipelined<16, 2><<<grid, block>>>(d_A, d_B, d_C, N);
    }
}

int main() {
    std::cout << "🚀 CUDA Matrix Multiplication Comparison (CUDA Events Timing)\n";
    std::cout << "============================================================\n\n";

    const int N = 1024;
    const size_t matrix_bytes = N * N * sizeof(float);

    std::cout << "Matrix size: " << N << "x" << N << " (" << matrix_bytes / (1024*1024) << " MB per matrix)\n";

    constexpr int TILE_WIDTH = 16;
    constexpr int NUM_STAGES = 2;
    constexpr int NAIVE_BLOCK_SIZE = 16;

    std::cout << "Tile width: " << TILE_WIDTH << "\n";
    std::cout << "Pipeline stages: " << NUM_STAGES << "\n";
    std::cout << "Naive block size: " << NAIVE_BLOCK_SIZE << "x" << NAIVE_BLOCK_SIZE << "\n\n";

    std::vector<float> h_A(N * N), h_B(N * N);
    std::vector<float> h_C_naive(N * N), h_C_tiled(N * N), h_C_pipelined(N * N);

    std::cout << "🔄 Initializing matrices...\n";
    initializeMatrix(h_A, N, 42);
    initializeMatrix(h_B, N, 123);

    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, matrix_bytes), "Failed to allocate d_A");
    checkCudaError(cudaMalloc(&d_B, matrix_bytes), "Failed to allocate d_B");
    checkCudaError(cudaMalloc(&d_C, matrix_bytes), "Failed to allocate d_C");

    std::cout << "📤 Copying matrices to GPU...\n";
    checkCudaError(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice),
                  "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice),
                  "Failed to copy B to device");

    dim3 naive_block(NAIVE_BLOCK_SIZE, NAIVE_BLOCK_SIZE);
    dim3 naive_grid((N + NAIVE_BLOCK_SIZE - 1) / NAIVE_BLOCK_SIZE,
                   (N + NAIVE_BLOCK_SIZE - 1) / NAIVE_BLOCK_SIZE);

    dim3 tiled_block(TILE_WIDTH, TILE_WIDTH);
    dim3 tiled_grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                   (N + TILE_WIDTH - 1) / TILE_WIDTH);

    std::cout << "Naive grid: " << naive_grid.x << "x" << naive_grid.y << " blocks\n";
    std::cout << "Tiled grid: " << tiled_grid.x << "x" << tiled_grid.y << " blocks\n\n";

    std::cout << "⚡ Benchmarking naive matrix multiplication (CUDA Events)...\n";
    double naive_time = benchmarkKernelWithEvents([&]() {
        launch_naive_kernel(d_A, d_B, d_C, N, naive_grid, naive_block);
    });
    checkCudaError(cudaMemcpy(h_C_naive.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost),
                  "Failed to copy naive result");

    std::cout << "🔧 Benchmarking tiled matrix multiplication (CUDA Events)...\n";
    double tiled_time = benchmarkKernelWithEvents([&]() {
        launch_tiled_kernel(d_A, d_B, d_C, N, tiled_grid, tiled_block, TILE_WIDTH);
    });
    checkCudaError(cudaMemcpy(h_C_tiled.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost),
                  "Failed to copy tiled result");

    std::cout << "🚀 Benchmarking pipelined matrix multiplication (CUDA Events)...\n";
    double pipelined_time = benchmarkKernelWithEvents([&]() {
        launch_pipelined_kernel(d_A, d_B, d_C, N, tiled_grid, tiled_block, TILE_WIDTH, NUM_STAGES);
    });
    checkCudaError(cudaMemcpy(h_C_pipelined.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost),
                  "Failed to copy pipelined result");

    std::cout << "\n🔍 Verifying results...\n";
    bool tiled_vs_naive = verifyResults(h_C_naive, h_C_tiled, N);
    bool pipelined_vs_naive = verifyResults(h_C_naive, h_C_pipelined, N);

    std::cout << "\n📊 Performance Results (CUDA Events Timing):\n";
    std::cout << "==============================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Naive kernel:     " << naive_time << " ms\n";
    std::cout << "Tiled kernel:     " << tiled_time << " ms\n";
    std::cout << "Pipelined kernel: " << pipelined_time << " ms\n\n";

    double tiled_speedup = naive_time / tiled_time;
    double pipelined_speedup = naive_time / pipelined_time;
    double pipeline_vs_tiled = tiled_time / pipelined_time;

    std::cout << "Speedup Analysis:\n";
    std::cout << "Tiled vs Naive:      " << tiled_speedup << "x\n";
    std::cout << "Pipelined vs Naive:  " << pipelined_speedup << "x\n";
    std::cout << "Pipelined vs Tiled:  " << pipeline_vs_tiled << "x\n\n";

    double total_ops = 2.0 * N * N * N;
    double naive_gflops = total_ops / (naive_time * 1e-3) / 1e9;
    double tiled_gflops = total_ops / (tiled_time * 1e-3) / 1e9;
    double pipelined_gflops = total_ops / (pipelined_time * 1e-3) / 1e9;

    std::cout << "GFLOPS Performance:\n";
    std::cout << "Naive:     " << naive_gflops << " GFLOPS\n";
    std::cout << "Tiled:     " << tiled_gflops << " GFLOPS\n";
    std::cout << "Pipelined: " << pipelined_gflops << " GFLOPS\n\n";

    double memory_ops = 3.0 * matrix_bytes;
    double naive_bandwidth = memory_ops / (naive_time * 1e-3) / (1024*1024*1024);
    double tiled_bandwidth = memory_ops / (tiled_time * 1e-3) / (1024*1024*1024);
    double pipelined_bandwidth = memory_ops / (pipelined_time * 1e-3) / (1024*1024*1024);

    std::cout << "Memory Bandwidth:\n";
    std::cout << "Naive:     " << naive_bandwidth << " GB/s\n";
    std::cout << "Tiled:     " << tiled_bandwidth << " GB/s\n";
    std::cout << "Pipelined: " << pipelined_bandwidth << " GB/s\n\n";

    std::cout << "✅ Verification Results:\n";
    std::cout << "Tiled vs Naive:      " << (tiled_vs_naive ? "PASSED" : "FAILED") << "\n";
    std::cout << "Pipelined vs Naive:  " << (pipelined_vs_naive ? "PASSED" : "FAILED") << "\n\n";


    if (pipelined_speedup > tiled_speedup * 1.1) {
        std::cout << "🎉 Pipelining shows clear benefits for this problem size!\n";
    } else {
        std::cout << "📝 Pipelining benefits may be more apparent with larger matrices\n";
        std::cout << "   or on newer GPU architectures with better async support.\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\n🎯 Benchmark completed successfully with CUDA Events timing!\n";

    return 0;
}

extern "C" void solution(const float* d_A, const float* d_B, float* d_C, size_t N) {
    constexpr int TILE_WIDTH = 16;
    constexpr int NUM_STAGES = 2;

    const int grid_dim = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(grid_dim, grid_dim);

    gemm_pipelined<TILE_WIDTH, NUM_STAGES><<<grid, block>>>(
        const_cast<float*>(d_A),
        const_cast<float*>(d_B),
        d_C,
        static_cast<int>(N)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

