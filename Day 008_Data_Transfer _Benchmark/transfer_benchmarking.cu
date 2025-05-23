#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstring>

#define CHECK_CUDA(call) \
    if (call != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(call) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define DATA_SIZE_MB 100
#define DATA_SIZE_BYTES (DATA_SIZE_MB * 1024 * 1024)

struct BenchmarkResult {
    std::string name;
    float time_ms;
    float bandwidth_gbps;
    
    BenchmarkResult(const std::string& n, float t, float b) : name(n), time_ms(t), bandwidth_gbps(b) {}
};

float benchmarkMemcpy(void* dst, void* src, size_t size, cudaMemcpyKind kind, bool async = false, cudaStream_t stream = 0) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start, stream));
    if (async) {
        CHECK_CUDA(cudaMemcpyAsync(dst, src, size, kind, stream));
    } else {
        CHECK_CUDA(cudaMemcpy(dst, src, size, kind));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms;
}

// 1. Host-to-Host (CPU copy)
BenchmarkResult testHostToHost() {
    char* h_src = (char*)malloc(DATA_SIZE_BYTES);
    char* h_dst = (char*)malloc(DATA_SIZE_BYTES);
    
    // Initialize memory to prevent optimization
    memset(h_src, 1, DATA_SIZE_BYTES);
    
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(h_dst, h_src, DATA_SIZE_BYTES);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    float ms = diff.count() * 1000;
    float bandwidth = (DATA_SIZE_BYTES / 1e9) / diff.count();
    
    free(h_src);
    free(h_dst);
    
    return BenchmarkResult("Host-to-Host (memcpy)", ms, bandwidth);
}

// 2. Basic GPU transfers
std::vector<BenchmarkResult> testBasicTransfers() {
    std::vector<BenchmarkResult> results;
    
    void* h_src; void* h_dst;
    CHECK_CUDA(cudaMallocHost(&h_src, DATA_SIZE_BYTES)); // Pinned host memory
    CHECK_CUDA(cudaMallocHost(&h_dst, DATA_SIZE_BYTES));

    void* d_src; void* d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, DATA_SIZE_BYTES));
    CHECK_CUDA(cudaMalloc(&d_dst, DATA_SIZE_BYTES));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Synchronous transfers
    float ms = benchmarkMemcpy(d_src, h_src, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
    results.emplace_back("H2D sync (pinned)", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    ms = benchmarkMemcpy(h_dst, d_src, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);
    results.emplace_back("D2H sync (pinned)", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    ms = benchmarkMemcpy(d_dst, d_src, DATA_SIZE_BYTES, cudaMemcpyDeviceToDevice);
    results.emplace_back("D2D sync", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    // Asynchronous transfers
    ms = benchmarkMemcpy(d_src, h_src, DATA_SIZE_BYTES, cudaMemcpyHostToDevice, true, stream);
    results.emplace_back("H2D async (pinned)", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    ms = benchmarkMemcpy(h_dst, d_src, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost, true, stream);
    results.emplace_back("D2H async (pinned)", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    ms = benchmarkMemcpy(d_dst, d_src, DATA_SIZE_BYTES, cudaMemcpyDeviceToDevice, true, stream);
    results.emplace_back("D2D async", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return results;
}

// 3. Pinned vs Pageable memory comparison
std::vector<BenchmarkResult> testPinnedVsPageable() {
    std::vector<BenchmarkResult> results;
    
    void* d_mem, *h_pinned, *h_pageable;
    CHECK_CUDA(cudaMalloc(&d_mem, DATA_SIZE_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_pinned, DATA_SIZE_BYTES));
    h_pageable = malloc(DATA_SIZE_BYTES);

    float ms = benchmarkMemcpy(h_pageable, d_mem, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);
    results.emplace_back("D2H pageable", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    ms = benchmarkMemcpy(h_pinned, d_mem, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);
    results.emplace_back("D2H pinned", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    ms = benchmarkMemcpy(d_mem, h_pageable, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
    results.emplace_back("H2D pageable", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    ms = benchmarkMemcpy(d_mem, h_pinned, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
    results.emplace_back("H2D pinned", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));

    CHECK_CUDA(cudaFree(d_mem));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    free(h_pageable);

    return results;
}

// 4. Unified Memory test
BenchmarkResult testUnifiedMemory() {
    float* managed_mem;
    CHECK_CUDA(cudaMallocManaged(&managed_mem, DATA_SIZE_BYTES));
    
    // Force memory to be allocated on device
    CHECK_CUDA(cudaMemPrefetchAsync(managed_mem, DATA_SIZE_BYTES, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float ms = benchmarkMemcpy(managed_mem, managed_mem, DATA_SIZE_BYTES, cudaMemcpyDeviceToDevice);
    
    CHECK_CUDA(cudaFree(managed_mem));
    
    return BenchmarkResult("Unified Memory Access", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));
}

// 5. Mapped Memory test
BenchmarkResult testMappedMemory() {
    void* h_mapped;
    void* d_mapped;
    
    CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    CHECK_CUDA(cudaHostAlloc(&h_mapped, DATA_SIZE_BYTES, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_mapped, h_mapped, 0));

    float ms = benchmarkMemcpy(d_mapped, d_mapped, DATA_SIZE_BYTES, cudaMemcpyDeviceToDevice);

    CHECK_CUDA(cudaFreeHost(h_mapped));
    
    return BenchmarkResult("Mapped Memory Access", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));
}

// 6. Multiple Streams test
BenchmarkResult testMultipleStreams() {
    const int num_streams = 4;
    const size_t chunk_size = DATA_SIZE_BYTES / num_streams;

    char* h_src, *h_dst;
    char* d_src, *d_dst;
    CHECK_CUDA(cudaMallocHost(&h_src, DATA_SIZE_BYTES));
    CHECK_CUDA(cudaMallocHost(&h_dst, DATA_SIZE_BYTES));
    CHECK_CUDA(cudaMalloc(&d_src, DATA_SIZE_BYTES));
    CHECK_CUDA(cudaMalloc(&d_dst, DATA_SIZE_BYTES));

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d_dst + i * chunk_size, d_src + i * chunk_size, 
                                   chunk_size, cudaMemcpyDeviceToDevice, streams[i]));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    for (int i = 0; i < num_streams; ++i)
        CHECK_CUDA(cudaStreamDestroy(streams[i]));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));

    return BenchmarkResult("D2D Multi-Stream (4 streams)", ms, (DATA_SIZE_MB / 1024.0f) / (ms / 1000.0f));
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "+" << std::string(28, '-') << "+" << std::string(12, '-') << "+" << std::string(15, '-') << "+\n";
    std::cout << "| " << std::setw(26) << std::left << "Transfer Type"
              << " | " << std::setw(10) << "Time (ms)"
              << " | " << std::setw(13) << "Bandwidth (GB/s)" << " |\n";
    std::cout << "+" << std::string(28, '-') << "+" << std::string(12, '-') << "+" << std::string(15, '-') << "+\n";
    
    for (const auto& result : results) {
        std::cout << "| " << std::setw(26) << std::left << result.name
                  << " | " << std::setw(10) << std::right << result.time_ms
                  << " | " << std::setw(13) << std::right << result.bandwidth_gbps << " |\n";
    }
    
    std::cout << "+" << std::string(28, '-') << "+" << std::string(12, '-') << "+" << std::string(15, '-') << "+\n";
}

int main() {
    std::cout << "\n=== CUDA Memory Transfer Comprehensive Benchmark ===\n";
    std::cout << "Data Size: " << DATA_SIZE_MB << " MB\n\n";

    // Get GPU information
    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "Peak Memory Bandwidth: " << std::fixed << std::setprecision(1) 
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s\n\n";

    std::vector<BenchmarkResult> allResults;

    // Run all benchmarks
    std::cout << "Running benchmarks...\n";
    
    allResults.push_back(testHostToHost());
    
    auto basicResults = testBasicTransfers();
    allResults.insert(allResults.end(), basicResults.begin(), basicResults.end());
    
    auto pinnedResults = testPinnedVsPageable();
    allResults.insert(allResults.end(), pinnedResults.begin(), pinnedResults.end());
    
    allResults.push_back(testUnifiedMemory());
    allResults.push_back(testMappedMemory());
    allResults.push_back(testMultipleStreams());

    // Print comprehensive results table
    std::cout << "\n=== COMPREHENSIVE RESULTS ===\n";
    printResults(allResults);

    // Performance analysis
    std::cout << "\n=== PERFORMANCE ANALYSIS ===\n";
    
    // Find best performing transfers
    float best_h2d = 0, best_d2h = 0, best_d2d = 0;
    std::string best_h2d_name, best_d2h_name, best_d2d_name;
    
    for (const auto& result : allResults) {
        if (result.name.find("H2D") != std::string::npos && result.bandwidth_gbps > best_h2d) {
            best_h2d = result.bandwidth_gbps;
            best_h2d_name = result.name;
        }
        if (result.name.find("D2H") != std::string::npos && result.bandwidth_gbps > best_d2h) {
            best_d2h = result.bandwidth_gbps;
            best_d2h_name = result.name;
        }
        if (result.name.find("D2D") != std::string::npos && result.bandwidth_gbps > best_d2d) {
            best_d2d = result.bandwidth_gbps;
            best_d2d_name = result.name;
        }
    }
    
    std::cout << "Best Host-to-Device:   " << best_h2d_name << " (" << best_h2d << " GB/s)\n";
    std::cout << "Best Device-to-Host:   " << best_d2h_name << " (" << best_d2h << " GB/s)\n";
    std::cout << "Best Device-to-Device: " << best_d2d_name << " (" << best_d2d << " GB/s)\n";

    return 0;
}
