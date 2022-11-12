// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE       16

//@@ insert code here
__global__ void float2unsignedchar(float *input, unsigned char *output, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len){
        output[i] = (unsigned char)(255 * input[i]);
    }
}

__global__ void rgb2grayscale(unsigned char *input, unsigned char *output, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len){
        unsigned char r = input[3 * i];
        unsigned char g = input[3 * i + 1];
        unsigned char b = input[3 * i + 2];
        output[i] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

__global__ void histogram(unsigned char *input, unsigned int *output, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned int histo[HISTOGRAM_LENGTH];
    if (threadIdx.x < HISTOGRAM_LENGTH){
        histo[threadIdx.x] = 0;
    }
    __syncthreads();
    if (i < len){
        atomicAdd(&(histo[input[i]]), 1); // subtotal in each block
    }
    __syncthreads();
    if (threadIdx.x < HISTOGRAM_LENGTH){
        atomicAdd(&(output[threadIdx.x]), histo[threadIdx.x]); // add the subtotal to the global memory
    }
}

__global__ void computeCDF(unsigned int *input, float *output, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float cdf[HISTOGRAM_LENGTH];
    if (i < HISTOGRAM_LENGTH){
        cdf[i] = input[i];
    }
    __syncthreads();
    for (int stride = 1; stride < HISTOGRAM_LENGTH; stride *= 2){
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < HISTOGRAM_LENGTH){
            cdf[index] += cdf[index - stride];
        }
    }
    for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2){
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < HISTOGRAM_LENGTH){
            cdf[index + stride] += cdf[index];
        }
    }
    __syncthreads();
    if (i < HISTOGRAM_LENGTH){
        output[i] = (float)cdf[i] * 1.0 / len;
    }
}

__global__ void equalize(unsigned char *image, float *cdf, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len){
        float x = 255.0 * (cdf[image[i]] - cdf[0]) / (1 - cdf[0]);
        float correctColor = min(max(x, 0.0f), 255.0f);
        image[i] = (unsigned char)correctColor;
    }
}

__global__ void unsignedchar2float(unsigned char *input, float *output, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len){
        output[i] = (float)input[i] * 1.0 / 255;
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInoutImageData;
  unsigned char *deviceDataChar;
  unsigned char *deviceGrayscale;
  unsigned int *deviceHistogram;
  float *deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceInoutImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceDataChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGrayscale, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInoutImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch the kernel
  dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 dimGrid((imageWidth * imageHeight * imageChannels + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE), 1, 1);
  float2unsignedchar<<<dimGrid, dimBlock>>>(deviceInoutImageData, deviceDataChar, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  dim3 dimBlock2(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 dimGrid2((imageWidth * imageHeight + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE), 1, 1);
  rgb2grayscale<<<dimGrid2, dimBlock2>>>(deviceDataChar, deviceGrayscale, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  histogram<<<dimGrid2, dimBlock2>>>(deviceGrayscale, deviceHistogram, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  computeCDF<<<1, HISTOGRAM_LENGTH>>>(deviceHistogram, deviceCDF, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  equalize<<<dimGrid, dimBlock>>>(deviceDataChar, deviceCDF, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  unsignedchar2float<<<dimGrid, dimBlock>>>(deviceDataChar, deviceInoutImageData, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  wbTime_start(Copy, "Copying output memory to the CPU.");
  cudaMemcpy(hostOutputImageData, deviceInoutImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU.");

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInoutImageData);
  cudaFree(deviceDataChar);
  cudaFree(deviceGrayscale);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);

  return 0;
}
