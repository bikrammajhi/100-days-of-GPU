#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * BLUR_SIZE defines the radius of the blur kernel
 * For BLUR_SIZE=1, we use a 3x3 kernel (2*BLUR_SIZE+1 in each dimension)
 * Common errors: Using incorrect bounds in the kernel loops based on this value
 */
#define BLUR_SIZE 1

/**
 * CUDA kernel for image blurring
 * Each thread processes one output pixel by averaging its neighborhood
 *
 * @param image      Input image data (grayscale)
 * @param blurred    Output blurred image
 * @param width      Image width
 * @param height     Image height
 * 
 * Common errors:
 * 1. Incorrect loop bounds (using < instead of <= or vice versa)
 * 2. Forgetting to check boundary conditions
 * 3. Incorrect calculation of thread indices
 * 4. Incorrect calculation of average value (division)
 */
__global__ void blurKernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
  // Calculate the row and column for this thread
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;

  // Only process pixels within the image bounds
  if (outRow < height && outCol < width) {
    unsigned int sum = 0;
    unsigned int pixelsUsed = 0; // For a proper average calculation (alternative approach)

    // Loop through the blur window - note we use +1 in bound to include the right/bottom pixels
    // This ensures we process (2*BLUR_SIZE+1)x(2*BLUR_SIZE+1) pixels
    for (int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow) {
      for (int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1; ++inCol) {
        // Only include pixels that are within the image bounds
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          sum += image[inRow * width + inCol];
          pixelsUsed++; // Count actual pixels used (alternative approach)
        }
      }
    }

    // Calculate the blurred pixel value
    // Using the theoretical count - (2*BLUR_SIZE+1)^2
    unsigned int count = (2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1);
    
    // Alternative approach: Use actual pixels counted (better for edge cases)
    // blurred[outRow * width + outCol] = (unsigned char)(sum / pixelsUsed);
    
    // Using theoretical count (works well when all pixels in window are valid)
    blurred[outRow * width + outCol] = (unsigned char)(sum / count);
  }
}

/**
 * Host function to manage the blur operation on GPU
 * 
 * @param image      Input image data (grayscale)
 * @param blurred    Output blurred image
 * @param width      Image width
 * @param height     Image height
 * 
 * Common errors:
 * 1. Incorrect memory allocation/deallocation
 * 2. Not checking CUDA API call results
 * 3. Using wrong cudaMemcpy direction
 * 4. Miscalculating grid/block dimensions
 */
void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){
    // CUDA event objects for timing measurements
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Device pointers
    unsigned char* image_d;
    unsigned char* blurred_d;

    printf("Starting GPU blur operation...\n");
    
    // Record the start time
    cudaEventRecord(start);

    // Allocate memory in GPU
    printf("Allocating GPU memory...\n");
    cudaError_t error;
    error = cudaMalloc((void**)&image_d, width * height * sizeof(unsigned char));
    if (error != cudaSuccess) {
        printf("Error allocating device memory for image: %s\n", cudaGetErrorString(error));
        return;
    }
    
    error = cudaMalloc((void**)&blurred_d, width * height * sizeof(unsigned char));
    if (error != cudaSuccess) {
        printf("Error allocating device memory for blurred image: %s\n", cudaGetErrorString(error));
        cudaFree(image_d);
        return;
    }

    // Record memory allocation time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Memory allocation time: %.2f ms\n", milliseconds);
    cudaEventRecord(start);

    // Transfer data to GPU
    printf("Copying data to GPU...\n");
    error = cudaMemcpy(image_d, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error copying data to device: %s\n", cudaGetErrorString(error));
        cudaFree(image_d);
        cudaFree(blurred_d);
        return;
    }

    // Record data transfer time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Host to device transfer time: %.2f ms\n", milliseconds);
    cudaEventRecord(start);

    // Call Kernel
    printf("Launching kernel...\n");
    // Define the thread block and grid dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    printf("Grid size: %d x %d blocks\n", numBlocks.x, numBlocks.y);
    printf("Block size: %d x %d threads\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Image size: %d x %d pixels\n", width, height);

    // Launch the kernel
    blurKernel<<<numBlocks, threadsPerBlock>>>(image_d, blurred_d, width, height);
    
    // Check for kernel launch errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        cudaFree(image_d);
        cudaFree(blurred_d);
        return;
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Check for kernel execution errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(error));
        cudaFree(image_d);
        cudaFree(blurred_d);
        return;
    }

    // Record kernel execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.2f ms\n", milliseconds);
    cudaEventRecord(start);

    // Transfer data back to CPU
    printf("Copying results back to host...\n");
    error = cudaMemcpy(blurred, blurred_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error copying data back to host: %s\n", cudaGetErrorString(error));
        cudaFree(image_d);
        cudaFree(blurred_d);
        return;
    }

    // Record device to host transfer time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Device to host transfer time: %.2f ms\n", milliseconds);

    // Free memory in GPU
    printf("Freeing GPU memory...\n");
    cudaFree(image_d);
    cudaFree(blurred_d);
    
    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("GPU blur operation completed successfully.\n");
}

/**
 * Print properties of all available CUDA-capable devices
 * 
 * Common errors:
 * 1. Not checking for CUDA device existence
 * 2. Not handling errors from CUDA API calls
 */
void print_cuda_device_info() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("Error getting CUDA device count: %s\n", cudaGetErrorString(error));
        return;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return;
    }

    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, i);
        
        if (error != cudaSuccess) {
            printf("Error getting device properties: %s\n", cudaGetErrorString(error));
            continue;
        }

        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f GB\n",
               (float)deviceProp.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max threads dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Clock rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        
        // Additional helpful information
        printf("  L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
        printf("  Memory Clock Rate: %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.1f GB/s\n",
               2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
    }  
}

/**
 * Main function demonstrating the use of CUDA for image blurring
 * 
 * Common errors:
 * 1. Not checking for memory allocation failures
 * 2. Not freeing allocated memory
 * 3. Not checking device availability
 */
int main(int argc, char** argv) {
    // Overall timing
    cudaEvent_t totalStart, totalStop;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);
    float totalTime = 0;
    
    cudaEventRecord(totalStart);
    
    // Check if CUDA device is available
    printf("Checking CUDA devices...\n");
    print_cuda_device_info();
    
    // Example usage of blur function
    const unsigned int width = 1024;
    const unsigned int height = 768;
    
    printf("\nPreparing image of size %dx%d...\n", width, height);
    
    // Allocate memory for input and output images
    unsigned char* image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (image == NULL) {
        printf("Error: Failed to allocate host memory for input image\n");
        return -1;
    }
    
    unsigned char* blurred = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (blurred == NULL) {
        printf("Error: Failed to allocate host memory for output image\n");
        free(image);
        return -1;
    }
    
    // Initialize the input image with some pattern (e.g., gradient)
    printf("Initializing image with gradient pattern...\n");
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            image[i * width + j] = (i + j) % 256;
        }
    }
    
    // Run the blur kernel
    printf("\nRunning GPU blur operation...\n");
    blur_gpu(image, blurred, width, height);
    
    // Record total execution time
    cudaEventRecord(totalStop);
    cudaEventSynchronize(totalStop);
    cudaEventElapsedTime(&totalTime, totalStart, totalStop);
    printf("\nTotal execution time: %.2f ms\n", totalTime);
    
    // Print a small sample of the result for verification
    printf("\nSample of original image:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%3d ", image[i * width + j]);
        }
        printf("\n");
    }
    
    printf("\nSample of blurred image:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%3d ", blurred[i * width + j]);
        }
        printf("\n");
    }
    
    
    // Free memory
    printf("\nCleaning up resources...\n");
    free(image);
    free(blurred);
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalStop);
    
    // Reset CUDA device to clear any remaining resources
    cudaDeviceReset();
    
    printf("Program completed successfully.\n");
    return 0;
}