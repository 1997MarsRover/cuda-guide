#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

// Structure to increase memory usage per element
struct WorkloadData {
    float data[16];  // 64 bytes per structure
};

// Kernel with memory-intensive operations
__global__ void stressKernel(WorkloadData* data, WorkloadData* output, size_t n, int intensity_level) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        WorkloadData work = data[idx];
        
        // Process each element in the structure
        for (int j = 0; j < 16; j++) {
            float x = work.data[j];
            // Computational work
            for (int i = 0; i < intensity_level * 100; i++) {
                x = sinf(x) * sqrtf(x * x + 1.0f) + cosf(x);
                if (intensity_level > 3) {
                    x = tanf(x) * logf(fabsf(x) + 1.0f);
                }
            }
            work.data[j] = x;
        }
        
        output[idx] = work;
    }
}

int main(int argc, char** argv) {
    // Default intensity level (1-5)
    int intensity_level = 3;
    if (argc > 1) {
        intensity_level = atoi(argv[1]);
        if (intensity_level < 1) intensity_level = 1;
        if (intensity_level > 5) intensity_level = 5;
    }
    
    // Calculate memory sizes for approximately 10GB total GPU memory usage
    const size_t target_memory = (size_t)10 * 1024 * 1024 * 1024; // 10 GB
    const size_t struct_size = sizeof(WorkloadData);
    const size_t num_elements = target_memory / (2 * struct_size); // Divide by 2 because we need two buffers
    const size_t total_size = num_elements * struct_size;
    
    printf("Starting GPU stress test with high memory usage:\n");
    printf("Memory per buffer: %.2f GB\n", (float)total_size / (1024*1024*1024));
    printf("Total GPU memory usage: %.2f GB\n", (float)(total_size * 2) / (1024*1024*1024));
    printf("Number of elements: %zu\n", num_elements);
    printf("Intensity level: %d\n", intensity_level);
    
    // Allocate host memory
    WorkloadData* h_data = (WorkloadData*)malloc(total_size);
    if (!h_data) {
        printf("Failed to allocate host memory!\n");
        return 1;
    }
    
    // Initialize data
    for (size_t i = 0; i < num_elements; i++) {
        for (int j = 0; j < 16; j++) {
            h_data[i].data[j] = (float)i + (float)j * 0.1f;
        }
    }
    
    // Allocate device memory
    WorkloadData *d_data, *d_output;
    cudaError_t cuda_status;
    
    cuda_status = cudaMalloc(&d_data, total_size);
    if (cuda_status != cudaSuccess) {
        printf("cudaMalloc failed for d_data: %s\n", cudaGetErrorString(cuda_status));
        free(h_data);
        return 1;
    }
    
    cuda_status = cudaMalloc(&d_output, total_size);
    if (cuda_status != cudaSuccess) {
        printf("cudaMalloc failed for d_output: %s\n", cudaGetErrorString(cuda_status));
        cudaFree(d_data);
        free(h_data);
        return 1;
    }
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Configure grid and block dimensions
    int threadsPerBlock = 256;  // Standard thread block size
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("GPU: %s\n", prop.name);
    printf("Grid size: %d blocks x %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // Test duration (30 minutes)
    const int TEST_DURATION_SECONDS = 1800;
    time_t start_time = time(NULL);
    int iterations = 0;
    
    // Main test loop
    while (difftime(time(NULL), start_time) < TEST_DURATION_SECONDS) {
        // Copy input data to device
        cudaEventRecord(start);
        cuda_status = cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            printf("cudaMemcpy to device failed: %s\n", cudaGetErrorString(cuda_status));
            break;
        }
        
        // Launch kernel
        stressKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output, num_elements, intensity_level);
        
        // Copy results back (verify memory access)
        cuda_status = cudaMemcpy(h_data, d_output, total_size, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            printf("cudaMemcpy from device failed: %s\n", cudaGetErrorString(cuda_status));
            break;
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate timing
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Check for errors
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            printf("Kernel execution failed: %s\n", cudaGetErrorString(cuda_status));
            break;
        }
        
        // Print progress
        int elapsed_seconds = (int)difftime(time(NULL), start_time);
        printf("Time: %d:%02d | Iteration: %d | Last iteration time: %.2f ms\n",
               elapsed_seconds / 60, elapsed_seconds % 60, ++iterations, milliseconds);
        
        cudaDeviceSynchronize();
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_output);
    free(h_data);
    
    printf("\nTest completed with %d iterations\n", iterations);
    return 0;
}