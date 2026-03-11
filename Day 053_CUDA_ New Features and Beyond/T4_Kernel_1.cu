#include <cuda.h>
#include <mma.h>

template <unsigned int BM_dim,
unsigned int BN_dim,
unsigned int BK_dim,
unsigned int WM_dim,
unsigned int WN_dim,
unsigned int WK_dim,
unsigned int NUM_THREADS>

__global__ void kernel_1(
    half* A,
    half* B,
    half* C,
    half* D,
    const float alpha,
    const float beta,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
){
    // Define the dimensions of the MMA matrix
    constexpr unsigned int MMA_M_dim = 16;
    constexpr unsigned int MMA_N_dim = 8;
    constexpr unsigned int MMA_K_dim = 8;

    // For index calculation
    const unsigned int A_stride = K;
    const unsigned int B_stride = N;
    const unsigned int CD_stride = N;

    // loop bounds, constexpr to avoid runtime calculations (loop unrolling)
    const unsigned int num_block_tiles_k = K / BK_dim;
    constexpr unsigned int warp_tiles_per_block_k = BK_dim / WK_dim;
    constexpr unsigned int mma_tiles_per_warp_k = WK_dim / MMA_K_dim;
    constexpr unsigned int mma_tiles_per_warp_n = WN_dim / MMA_N_dim;
    constexpr unsigned int mma_tiles_per_warp_m = WM_dim / MMA_M_dim;

    // Compute block/warp indices
    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;
    const unsigned int warp_m = threadIdx.y;
    // Compute the warp index in the x-dimension. Each warp contains 32 consecutive threads along threadIdx.x.
    const unsigned int warp_n = threadIdx.x/32;

    // shared memory allocation
    extern __shared__ half shared_mem[];
    half* A_block_smem = shared_mem;
    half* B_block_smem = &shared_mem[BM_dim * BK_dim];
    
    // declare register storage 
    // ptx instrustion expect uint32_t registers, where each uint32_t is 2 halfs packed together
    uint32_t acc_register[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2];

    // conveniecne cast to half for accumulator registers
    half (&acc_register)[mma_tiles_per_warp_m][mma_tiles_per_warp_n][2] = reinterpret_cast<half (&)[mma_tiles_per_warp_m]
    [mma_tiles_per_warp_n][4]>(acc_register);

    uint32_t A_register[mma_tiles_per_warp_m][mma_tiles_per_warp_k][2];
    uint32_t B_register[mma_tiles_per_warp_k][mma_tiles_per_warp_n];

    // accumulator start at zero
    for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
        for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++) {
            acc_register[mma_m][mma_n][0] = 0;
            acc_register[mma_m][mma_n][1] = 0;
            acc_register[mma_m][mma_n][2] = 0;
            acc_register[mma_m][mma_n][3] = 0;
    
        }
    }

    for (unsigned int block_k = 0; block_k < num_block_tiles_k; block_k++) {
        half* A_block = A + (block_m * BM_dim * A_stride) + (block_k * BK_dim);
        half* B_block = B + (block_k * BK_dim * B_stride) + (block_n * BN_dim);

        tileMemcpy(A_block_smem, A_block_gmem, K, BM_dim * BK_dim);
        tileMemcpy(B_block_smem, B_block_gmem, N, BK_dim * BN_dim);
        __syncthreads();

        for (unsigned int warp_k = 0; warp_k < warp_tiles_per_block_k; warp_k++) {
            
            
            half* A_warp_tile = A_block_smem + (warp_m * WM_dim * BM_dim) + (warp_k * WK_dim);
            half* B_warp_tile = B_block_smem + (warp_k * WK_dim * BK_dim) + (warp_n * WN_dim);
            uint32_t A_warp_tile_byte_offset = cvta_to_shared_u32(A_warp_tile);
            uint32_t B_warp_tile_byte_offset = cvta_to_shared_u32(B_warp_tile);

            // preload tiles of a into registers
            for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++) {
                    for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++) {
                        // byte offsett to the top left element of the mma tile
                        const unsigned int mma_tile_byte_offset = ((mma_m * MMA_M_dim * BK_dim) + (mma_k * MMA_K_dim)) * sizeof(half);

                        // byte offset to the start of this thread's half in the mma tile
                        const unsigned int thread_byte_offset = (threadIdx.x % MMA_M_dim) * BK_dim * sizeof(half);

                        // calculate offset in butes WRT to the start our share memeory allocation 
                        const unsigned int shared_mem_offset = A_warp_tile_byte_offset + mma_tile_byte_offset + thread_byte_offset;

                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
                            "{%0, %1}. [%2];"
                            : "=r"(A_register[mma_m][mma_k][0]), "=r"(A_register[mma_m][mma_k][1])
                            : "r"(thread_offset_bytes)
                        );

                    }
            }

            // preload tiles of b into registers
            for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++)
            {
                for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++)
                {
                    const unsigned int mma_tiles_byte_offset = ((mma_k * MMA_K_dim * BN_dim) + (mma_n * MMA_N_dim)) * sizeof(half);
                    const unsigned int thread_byte_offset = (threadIdx.x % MMA_K_dim) * BN_dim * sizeof(half);
                    const unsigned int thread_offset_bytes = B_warp_tile_byte_offset + mma_tiles_byte_offset + thread_byte_offset;

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"
                        "{%0}, [%1];"
                        : "=r"(B_register[mma_k][mma_n])
                        : "r"(thread_offset_bytes)
                    );
                }
            }

            // outer product between mma tiles
            for(unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k++){
                for (unsigned int mma_n = 0; mma_n < mma_tiles_per_warp_n; mma_n++){
                    for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m++){
                        asm volatile(
                            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"
                            "{%0, %1},"
                            "{%2, %3},"
                            "{%4}, "
                            "{%5, %6};"
                            : "=r"(acc_register[mma_m][mma_n][0]), "=r"(acc_register[mma_m][mma_n][1])
                            : "r"(A_register[mma_m][mma_k][0]), "r"(A_register[mma_m][mma_k][1]),
                              "r"(B_register[mma_k][mma_n])
                              "r"(acc_register[mma_m][mma_n][0]), "r"(acc_register[mma_m][mma_n][1])
                        );
                    }
                }
            }
        }

        __syncthreads();
    }

    