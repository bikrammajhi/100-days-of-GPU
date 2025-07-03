// Pseudocode
__shared__ float block_a[2][block_a_size]
__shared__ float block_b[2][block_b_size]
float fragment_a[2][8]
float fragment_b[2][8]
float accumulator[8][8]

block_a[0] = load first block of matrix A
block_b[0] = load first block of matrix B
fragment_a[0] = load first fragment from block_a[0]
fragment_b[0] = load first fragment from block_b[0]

for (block_k=0; block_k<(K/ks-1); block_k++) {
    block_idx = block_k % 2
    block_prefetch_idx = (block_k+1) % 2
    // prefetch next blocks (Shared Memory Double buffering)
    block_a[block_prefetch_idx] = load next block of matrix A
    block_a[block_prefetch_idx] = load next block of matrix B
    for (int warp_k=0; warp_k<8; warp_k++) {
        frag_idx = warp_k % 2
        frag_prefetch_idx = (warp_k + 1) % 2
        // prefetch next fragments (Register Double buffering)
        fragment_a[frag_prefetch_idx] = load next fragment from block_a[block_idx]
        fragment_b[frag_prefetch_idx] = load next fragment from block_b[block_idx]
        // use fragments loaded in previous iteration to calculate matrix product
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += fragment_a[frag_idx][i] * fragment_b[frag_idx][j];
            }
        }
    }
    fragment_a[0] = load first fragment from block_a[block_prefetch_idx]
    fragment_b[0] = load first fragment from block_b[block_prefetch_idx]
}

// final update of the accumulator using last blocks
for (int warp_k=0; warp_k<8; warp_k++) {
    frag_idx = warp_k % 2
    frag_prefetch_idx = (warp_k + 1) % 2
    // prefetch next fragments (Register Double buffering)
    fragment_a[frag_prefetch_idx] = load next fragment from block_a[block_prefetch_idx]
    fragment_b[frag_prefetch_idx] = load next fragment from block_b[block_prefetch_idx]
    // use fragments loaded in previous iteration to calculate matrix product
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            accumulator[i][j] += fragment_a[frag_idx][i] * fragment_b[frag_idx][j];
        }
    }
}

// After completing the matrix multiplication C=A*B, we perform one final update to the accumulator
// to compute  C=alpha*A*B before storing the result back to global memory:
for (int i=0; i<8; i++) {
    for (int j=0; j<8; j++) {
        accumulator[i][j] *= alpha;
    }
}

store_to_global_memory(accumulator)