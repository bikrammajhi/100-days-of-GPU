#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to convert an RGB image to grayscale.
// Each thread computes one pixel of the grayscale image.
__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, 
                                unsigned char* gray, unsigned int width, unsigned int height) {
    // Calculate the row and column of the pixel this thread will handle
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the linear index for the 1D image arrays
    unsigned int index = row * width + col;

    // Ensure we are within bounds of the image
    if(row < height && col < width) {
        // Convert the pixel to grayscale using the standard luminosity method
        gray[index] = (unsigned char)(0.299f * red[index] + 
                                      0.587f * green[index] + 
                                      0.114f * blue[index]);
    }
}

// Function to perform grayscale conversion using CUDA
void rgb2grayscale(unsigned char* red, unsigned char* green, unsigned char* blue, 
                   unsigned char* gray, unsigned int width, unsigned int height){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  
    cudaEventCreate(&stop);

    // Allocate memory on the GPU for each color channel and the grayscale image
    cudaEventRecord(start);  // Start timing for allocation
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**) &red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**) &green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**) &blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**) &gray_d, width * height * sizeof(unsigned char));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_allocation = 0;
    cudaEventElapsedTime(&milliseconds_allocation, start, stop);
    printf("Allocation time: %f ms\n", milliseconds_allocation);

    // Copy input image data from host to device
    cudaEventRecord(start);
    cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_copytoGPU = 0;
    cudaEventElapsedTime(&milliseconds_copytoGPU, start, stop);
    printf("Copy to GPU time: %f ms\n", milliseconds_copytoGPU);

    // Define the number of threads per block (32x32 is a common choice for 2D data)
    dim3 numThreadsPerBlock(32, 32);
    // Calculate the number of blocks required to cover the image dimensions
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 
                   (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);

    // Launch the CUDA kernel
    cudaEventRecord(start);
    rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_kernel = 0;
    cudaEventElapsedTime(&milliseconds_kernel, start, stop);
    printf("Kernel time: %f ms\n", milliseconds_kernel);

    // Copy the resulting grayscale image back to host memory
    cudaEventRecord(start);
    cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_copyFromGPU = 0;
    cudaEventElapsedTime(&milliseconds_copyFromGPU, start, stop);
    printf("Copy from GPU time: %f ms\n", milliseconds_copyFromGPU);

    // Free GPU memory to avoid memory leaks
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);

    // Print total time taken by all CUDA operations
    printf("Total CUDA operation time: %f ms\n", 
            milliseconds_allocation + milliseconds_copytoGPU + 
            milliseconds_kernel + milliseconds_copyFromGPU);
}

// Helper function to allocate memory for an image buffer on the host
unsigned char* allocate_image_buffer(unsigned int width, unsigned int height) {
    return (unsigned char*)malloc(width * height * sizeof(unsigned char));
}

// Fills red, green, and blue channels with test pattern for visualization and debugging
void initialize_test_image(unsigned char* red, unsigned char* green, unsigned char* blue, 
                           unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            unsigned int index = i * width + j;
            red[index] = (i * 255) / height;     // Vertical gradient
            green[index] = (j * 255) / width;    // Horizontal gradient
            blue[index] = 128;                   // Constant blue channel
        }
    }
}

// Saves a grayscale image to a file in PGM format for viewing
void save_grayscale_image(const char* filename, unsigned char* gray, 
                          unsigned int width, unsigned int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    // Write header for PGM file (Portable Gray Map format)
    fprintf(fp, "P5\n%u %u\n255\n", width, height);
    
    // Write the pixel data
    fwrite(gray, sizeof(unsigned char), width * height, fp);
    
    fclose(fp);
    printf("Grayscale image saved to %s\n", filename);
}

// Print properties of all available CUDA-capable devices
void print_cuda_device_info() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("Error getting CUDA device count: %s\n", cudaGetErrorString(error));
        return;
    }
    
    printf("Found %d CUDA devices\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

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
    }
}

// Main program: sets up test image and performs RGB to grayscale conversion
int main(int argc, char** argv) {
    // Set default image dimensions (1024x1024)
    unsigned int width = 1024;
    unsigned int height = 1024;

    // Allow user to override dimensions from command line
    if (argc >= 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }

    // Print available CUDA device information
    print_cuda_device_info();

    printf("\nConverting RGB image of size %ux%u to grayscale using CUDA\n", width, height);

    // Allocate memory for RGB and grayscale image buffers
    unsigned char* h_red = allocate_image_buffer(width, height);
    unsigned char* h_green = allocate_image_buffer(width, height);
    unsigned char* h_blue = allocate_image_buffer(width, height);
    unsigned char* h_gray = allocate_image_buffer(width, height);

    // Check for successful allocation
    if (!h_red || !h_green || !h_blue || !h_gray) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Initialize test image with gradient pattern
    initialize_test_image(h_red, h_green, h_blue, width, height);

    // Perform CUDA-based grayscale conversion
    rgb2grayscale(h_red, h_green, h_blue, h_gray, width, height);

    // Save the grayscale image to a PGM file
    save_grayscale_image("output_gray.pgm", h_gray, width, height);

    // Free host-side image memory
    free(h_red);
    free(h_green);
    free(h_blue);
    free(h_gray);

    printf("Conversion completed successfully\n");

    return 0;
}
