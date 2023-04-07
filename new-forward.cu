#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_SIZE 16

static inline __attribute__((always_inline)) void checkCUDAError(cudaError_t error) {
    if(error != cudaSuccess){
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int Width_grid = ceil(1.0 * Width_out / TILE_SIZE);
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int map_idx = blockIdx.x;
    int height_idx = (blockIdx.y / Width_grid) * TILE_SIZE + threadIdx.y;
    int width_idx = (blockIdx.y % Width_grid) * TILE_SIZE + threadIdx.x;
    int batch_idx = blockIdx.z;
    float acc = 0.0f;
    for (int c = 0; c < Channel; c++){
        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                acc += in_4d(batch_idx, c, height_idx + p, width_idx + q) * mask_4d(map_idx, c, p, q);
            }
        }
    }
    if(height_idx < Height_out && width_idx < Width_out){
        out_4d(batch_idx, map_idx, height_idx, width_idx) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int in_size = Batch * Channel * Height * Width;
    int out_size = Batch * Map_out * Height_out * Width_out;
    int mask_size = Channel * Map_out * K * K;
    checkCUDAError(cudaMalloc((void **) device_output_ptr, out_size * sizeof(float)));
    checkCUDAError(cudaMalloc((void **) device_input_ptr, in_size * sizeof(float)));
    checkCUDAError(cudaMalloc((void **) device_mask_ptr, mask_size * sizeof(float)));
    checkCUDAError(cudaMemcpy(*device_input_ptr, host_input, in_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCUDAError(cudaMemcpy(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice));

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int W_grid = ceil(1.0 * Width / TILE_SIZE);
    int H_grid = ceil(1.0 * Height / TILE_SIZE);
    int Y = H_grid * W_grid;
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim(Map_out, Y, Batch);
    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel)
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int out_size = Height_out * Width_out * Batch * Map_out;
    checkCUDAError(cudaMemcpy(host_output, device_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    checkCUDAError(cudaFree(device_output));
    checkCUDAError(cudaFree(device_input));
    checkCUDAError(cudaFree(device_mask));
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
