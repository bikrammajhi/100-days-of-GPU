#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Tile dimensions for shared memory
#define TILE_SIZE 32
#define THREADS_PER_BLOCK 256

// Matrix multiplication kernel using TMA
__global__ void matmul_tma_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b, 
    const __grid_constant__ CUtensorMap tensor_map_c,
    size_t m, size_t n, size_t k)
{
    // Shared memory buffers - 128 byte aligned for TMA
    __shared__ alignas(128) float smem_a[TILE_SIZE][TILE_SIZE];
    __shared__ alignas(128) float smem_b[TILE_SIZE][TILE_SIZE];
    __shared__ alignas(128) float smem_c[TILE_SIZE][TILE_SIZE];
    
    // Barriers for synchronizing TMA operations
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar_a, bar_b, bar_c;
    
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate global indices for this thread block
    int row = by * TILE_SIZE;
    int col = bx * TILE_SIZE;
    
    // Initialize barriers (only thread 0 does this)
    if (tx == 0 && ty == 0) {
        init(&bar_a, blockDim.x * blockDim.y);
        init(&bar_b, blockDim.x * blockDim.y);
        init(&bar_c, blockDim.x * blockDim.y);
        
        // Make barriers visible to TMA
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    
    // Initialize result tile to zero
    float result = 0.0f;
    
    // Loop over tiles in the K dimension
    for (int tile_k = 0; tile_k < (k + TILE_SIZE - 1) / TILE_SIZE; tile_k++) {
        barrier::arrival_token token_a, token_b;
        
        // Load tile from matrix A and B using TMA
        if (tx == 0 && ty == 0) {
            // Load A[row:row+TILE_SIZE, tile_k*TILE_SIZE:(tile_k+1)*TILE_SIZE]
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &smem_a, &tensor_map_a, 
                tile_k * TILE_SIZE,  // k offset
                row,                 // m offset
                bar_a
            );
            token_a = cuda::device::barrier_arrive_tx(bar_a, 1, sizeof(smem_a));
            
            // Load B[tile_k*TILE_SIZE:(tile_k+1)*TILE_SIZE, col:col+TILE_SIZE]  
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &smem_b, &tensor_map_b,
                col,                 // n offset  
                tile_k * TILE_SIZE,  // k offset
                bar_b
            );
            token_b = cuda::device::barrier_arrive_tx(bar_b, 1, sizeof(smem_b));
        } else {
            token_a = bar_a.arrive();
            token_b = bar_b.arrive();
        }
        
        // Wait for both tiles to arrive
        bar_a.wait(std::move(token_a));
        bar_b.wait(std::move(token_b));
        
        // Perform tile multiplication
        for (int i = 0; i < TILE_SIZE; i++) {
            // Bounds checking
            if (row + ty < m && tile_k * TILE_SIZE + i < k && col + tx < n) {
                result += smem_a[ty][i] * smem_b[i][tx];
            }
        }
        
        __syncthreads(); // Wait before loading next tiles
    }
    
    // Store result in shared memory first
    if (row + ty < m && col + tx < n) {
        smem_c[ty][tx] = result;
    } else {
        smem_c[ty][tx] = 0.0f; // Zero padding for out-of-bounds
    }
    
    // Make sure all results are written to shared memory
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    
    // Write result tile back to global memory using TMA
    barrier::arrival_token token_c;
    if (tx == 0 && ty == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(
            &tensor_map_c, 
            col,  // n offset
            row,  // m offset  
            &smem_c
        );
        
        // Wait for TMA to finish reading from shared memory
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
    
    // Cleanup barriers
    if (tx == 0 && ty == 0) {
        (&bar_a)->~barrier();
        (&bar_b)->~barrier(); 
        (&bar_c)->~barrier();
    }
}

// Helper function to get cuTensorMapEncodeTiled function pointer
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", 
        &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
    assert(driver_status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

// Helper function to create tensor map
CUtensorMap create_tensor_map(void* data_ptr, size_t height, size_t width) {
    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    
    uint64_t size[rank] = {width, height};  // Note: width first (fastest dimension)
    uint64_t stride[rank-1] = {width * sizeof(float)};
    uint32_t box_size[rank] = {TILE_SIZE, TILE_SIZE};
    uint32_t elem_stride[rank] = {1, 1};
    
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        data_ptr,
        size,
        stride, 
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    
    assert(res == CUDA_SUCCESS);
    return tensor_map;
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, 
                        size_t m, size_t n, size_t k) {
    
    // Create tensor maps for all matrices
    CUtensorMap map_a = create_tensor_map((void*)input_a, m, k);
    CUtensorMap map_b = create_tensor_map((void*)input_b, k, n);  
    CUtensorMap map_c = create_tensor_map((void*)output_c, m, n);
    
    // Calculate grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matmul_tma_kernel<<<gridDim, blockDim>>>(map_a, map_b, map_c, m, n, k);
    
    // Wait for completion
    cudaDeviceSynchronize();
}