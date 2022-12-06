#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH  16

__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int num_tile = ceil((float)(Width - K) / TILE_WIDTH);
    int h_out = ty + TILE_WIDTH * (bz / num_tile);
    int w_out = tx + TILE_WIDTH * (bz % num_tile);
    int m_out = by;  
    int b_out = bx;
    // const int shared_width = TILE_WIDTH + K - 1;
    // printf("K: %d", K);

    if (h_out < Height_out && w_out < Width_out) {
        float sum = 0;
        #pragma unroll
        for (int m_in = 0; m_in < Channel; m_in++) {
            sum += in_4d(b_out, m_in, h_out + 0, w_out + 0) * mask_4d(m_out, m_in, 0, 0)
            + in_4d(b_out, m_in, h_out + 0, w_out + 1) * mask_4d(m_out, m_in, 0, 1)
            + in_4d(b_out, m_in, h_out + 0, w_out + 2) * mask_4d(m_out, m_in, 0, 2)
            + in_4d(b_out, m_in, h_out + 0, w_out + 3) * mask_4d(m_out, m_in, 0, 3)
            + in_4d(b_out, m_in, h_out + 0, w_out + 4) * mask_4d(m_out, m_in, 0, 4)
            + in_4d(b_out, m_in, h_out + 0, w_out + 5) * mask_4d(m_out, m_in, 0, 5)
            + in_4d(b_out, m_in, h_out + 0, w_out + 6) * mask_4d(m_out, m_in, 0, 6)
            + in_4d(b_out, m_in, h_out + 1, w_out + 0) * mask_4d(m_out, m_in, 1, 0)
            + in_4d(b_out, m_in, h_out + 1, w_out + 1) * mask_4d(m_out, m_in, 1, 1)
            + in_4d(b_out, m_in, h_out + 1, w_out + 2) * mask_4d(m_out, m_in, 1, 2)
            + in_4d(b_out, m_in, h_out + 1, w_out + 3) * mask_4d(m_out, m_in, 1, 3)
            + in_4d(b_out, m_in, h_out + 1, w_out + 4) * mask_4d(m_out, m_in, 1, 4)
            + in_4d(b_out, m_in, h_out + 1, w_out + 5) * mask_4d(m_out, m_in, 1, 5)
            + in_4d(b_out, m_in, h_out + 1, w_out + 6) * mask_4d(m_out, m_in, 1, 6)
            + in_4d(b_out, m_in, h_out + 2, w_out + 0) * mask_4d(m_out, m_in, 2, 0)
            + in_4d(b_out, m_in, h_out + 2, w_out + 1) * mask_4d(m_out, m_in, 2, 1)
            + in_4d(b_out, m_in, h_out + 2, w_out + 2) * mask_4d(m_out, m_in, 2, 2)
            + in_4d(b_out, m_in, h_out + 2, w_out + 3) * mask_4d(m_out, m_in, 2, 3)
            + in_4d(b_out, m_in, h_out + 2, w_out + 4) * mask_4d(m_out, m_in, 2, 4)
            + in_4d(b_out, m_in, h_out + 2, w_out + 5) * mask_4d(m_out, m_in, 2, 5)
            + in_4d(b_out, m_in, h_out + 2, w_out + 6) * mask_4d(m_out, m_in, 2, 6)
            + in_4d(b_out, m_in, h_out + 3, w_out + 0) * mask_4d(m_out, m_in, 3, 0)
            + in_4d(b_out, m_in, h_out + 3, w_out + 1) * mask_4d(m_out, m_in, 3, 1)
            + in_4d(b_out, m_in, h_out + 3, w_out + 2) * mask_4d(m_out, m_in, 3, 2)
            + in_4d(b_out, m_in, h_out + 3, w_out + 3) * mask_4d(m_out, m_in, 3, 3)
            + in_4d(b_out, m_in, h_out + 3, w_out + 4) * mask_4d(m_out, m_in, 3, 4)
            + in_4d(b_out, m_in, h_out + 3, w_out + 5) * mask_4d(m_out, m_in, 3, 5)
            + in_4d(b_out, m_in, h_out + 3, w_out + 6) * mask_4d(m_out, m_in, 3, 6)
            + in_4d(b_out, m_in, h_out + 4, w_out + 0) * mask_4d(m_out, m_in, 4, 0)
            + in_4d(b_out, m_in, h_out + 4, w_out + 1) * mask_4d(m_out, m_in, 4, 1)
            + in_4d(b_out, m_in, h_out + 4, w_out + 2) * mask_4d(m_out, m_in, 4, 2)
            + in_4d(b_out, m_in, h_out + 4, w_out + 3) * mask_4d(m_out, m_in, 4, 3)
            + in_4d(b_out, m_in, h_out + 4, w_out + 4) * mask_4d(m_out, m_in, 4, 4)
            + in_4d(b_out, m_in, h_out + 4, w_out + 5) * mask_4d(m_out, m_in, 4, 5)
            + in_4d(b_out, m_in, h_out + 4, w_out + 6) * mask_4d(m_out, m_in, 4, 6)
            + in_4d(b_out, m_in, h_out + 5, w_out + 0) * mask_4d(m_out, m_in, 5, 0)
            + in_4d(b_out, m_in, h_out + 5, w_out + 1) * mask_4d(m_out, m_in, 5, 1)
            + in_4d(b_out, m_in, h_out + 5, w_out + 2) * mask_4d(m_out, m_in, 5, 2)
            + in_4d(b_out, m_in, h_out + 5, w_out + 3) * mask_4d(m_out, m_in, 5, 3)
            + in_4d(b_out, m_in, h_out + 5, w_out + 4) * mask_4d(m_out, m_in, 5, 4)
            + in_4d(b_out, m_in, h_out + 5, w_out + 5) * mask_4d(m_out, m_in, 5, 5)
            + in_4d(b_out, m_in, h_out + 5, w_out + 6) * mask_4d(m_out, m_in, 5, 6)
            + in_4d(b_out, m_in, h_out + 6, w_out + 0) * mask_4d(m_out, m_in, 6, 0)
            + in_4d(b_out, m_in, h_out + 6, w_out + 1) * mask_4d(m_out, m_in, 6, 1)
            + in_4d(b_out, m_in, h_out + 6, w_out + 2) * mask_4d(m_out, m_in, 6, 2)
            + in_4d(b_out, m_in, h_out + 6, w_out + 3) * mask_4d(m_out, m_in, 6, 3)
            + in_4d(b_out, m_in, h_out + 6, w_out + 4) * mask_4d(m_out, m_in, 6, 4)
            + in_4d(b_out, m_in, h_out + 6, w_out + 5) * mask_4d(m_out, m_in, 6, 5)
            + in_4d(b_out, m_in, h_out + 6, w_out + 6) * mask_4d(m_out, m_in, 6, 6);
        }
        out_4d(b_out, m_out, h_out, w_out) = sum;
    }

    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    cudaMalloc((void**)device_output_ptr, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1));
    cudaMalloc((void**)device_input_ptr, sizeof(float) * Batch * Channel * Height * Width);
    cudaMalloc((void**)device_mask_ptr, sizeof(float) * Map_out * Channel * K * K);
    // cudaMemcpy(*device_output_ptr, host_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Batch * Channel * Height * Width, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Map_out * Channel * K * K, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    dim3 dimGrid(Batch, Map_out, ceil((float)(Width - K) / TILE_WIDTH) * ceil((float)(Height - K) / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock, sizeof(float) * (TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
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
