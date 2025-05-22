// Matrices are stored in row-major order:
// Element at (row, col) is located at base pointer + row * width + col
typedef struct {
    int    width;    // Number of columns
    int    height;   // Number of rows
    float* elements; // Pointer to contiguous float data
} Matrix;

// Define the size of each thread block (16×16 threads)
#define BLOCK_SIZE 16

// Forward declaration of the GPU kernel for matrix multiplication
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

// Host-side matrix multiplication entry point
// Assumes A.width, A.height, B.width, B.height, C.width, and C.height
// are all multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // --------------------------------------------------
    // 1) Copy matrix A from host to device
    // --------------------------------------------------
    Matrix d_A;
    d_A.width  = A.width;
    d_A.height = A.height;
    size_t sizeA = A.width * A.height * sizeof(float);
    // Allocate device memory for A
    cudaMalloc(&d_A.elements, sizeA);
    // Copy A.elements (host) → d_A.elements (device)
    cudaMemcpy(d_A.elements, A.elements, sizeA,
               cudaMemcpyHostToDevice);

    // --------------------------------------------------
    // 2) Copy matrix B from host to device
    // --------------------------------------------------
    Matrix d_B;
    d_B.width  = B.width;
    d_B.height = B.height;
    size_t sizeB = B.width * B.height * sizeof(float);
    // Allocate device memory for B
    cudaMalloc(&d_B.elements, sizeB);
    // Copy B.elements (host) → d_B.elements (device)
    cudaMemcpy(d_B.elements, B.elements, sizeB,
               cudaMemcpyHostToDevice);

    // --------------------------------------------------
    // 3) Allocate space for matrix C on the device
    // --------------------------------------------------
    Matrix d_C;
    d_C.width  = C.width;
    d_C.height = C.height;
    size_t sizeC = C.width * C.height * sizeof(float);
    // Allocate device memory for result matrix C
    cudaMalloc(&d_C.elements, sizeC);

    // --------------------------------------------------
    // 4) Launch the CUDA kernel
    // --------------------------------------------------
    // Each block is BLOCK_SIZE×BLOCK_SIZE threads
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Grid dimensions: enough blocks to cover all rows and columns
    dim3 dimGrid(B.width  / dimBlock.x,
                 A.height / dimBlock.y);
    // Execute the kernel: computes C = A × B
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // --------------------------------------------------
    // 5) Copy the result matrix C back to the host
    // --------------------------------------------------
    cudaMemcpy(C.elements, d_C.elements, sizeC,
               cudaMemcpyDeviceToHost);

    // --------------------------------------------------
    // 6) Free all device memory
    // --------------------------------------------------
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// ----------------------------------------------------------------
// GPU kernel: multiplies A and B, writes result into C
// Each thread handles one element C[row, col]
// ----------------------------------------------------------------
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Accumulator for the dot product
    float Cvalue = 0.0f;

    // Compute global row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform the dot product of the row of A and column of B
    for (int e = 0; e < A.width; ++e) {
        // A[row, e] * B[e, col]
        float a_elem = A.elements[row * A.width + e];
        float b_elem = B.elements[e * B.width + col];
        Cvalue += a_elem * b_elem;
    }

    // Write the computed value into C[row, col]
    C.elements[row * C.width + col] = Cvalue;
}
