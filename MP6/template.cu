// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16
#define BLOCK_SIZE_SCAN 128

//@@ insert code here
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// BEGIN: Kernel to cast image to unsigned char
__global__ void cast_unsigned_char(float *input, unsigned char *output, int Width, int Height, int Channel){
    // Compute index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int w = bx * blockDim.x + tx; // Width
    int h = by * blockDim.y + ty; // Height
    int c = bz * blockDim.z + tz; // Channel; block index is unnecessary because there are 3 channels only

    if (w < Width && h < Height) {
        int pos = h * Width + w;
        int index = pos * Channel + c;
        output[index] = (unsigned char) (255. * input[index]);
    }
    __syncthreads();
}
// END: Kernel to cast image to unsigned char

// BEGIN: Kernel to convert image to grayscale
__global__ void to_grayscale(unsigned char *input, unsigned char *output, int Width, int Height, int Channel){
    // Compute index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int w = bx * blockDim.x + tx; // Width
    int h = by * blockDim.y + ty; // Height


    if (w < Width && h < Height) {
        int pos = h * Width + w;
        int index = pos * Channel;

        unsigned char r = input[index]; // R channel
        unsigned char g = input[index+1]; // G channel
        unsigned char b = input[index+2]; // B channel

        output[pos] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }

    __syncthreads();
}
// END: Kernel to convert image to grayscale

// BEGIN: Kernel to compute histogram
__global__ void compute_histogram(unsigned char *input, unsigned int *histogram, int Size) {
    // Use privatization
    __shared__ unsigned int private_histogram[HISTOGRAM_LENGTH];

    // Compute index
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int i = bx * blockDim.x + tx; // Linear index

    if (tx < HISTOGRAM_LENGTH) {
        private_histogram[tx] = 0;
    }

    __syncthreads();

    int stride = blockDim.x * gridDim.x;

    // Compute private histogram
    while (i < Size) {
        atomicAdd( &(private_histogram[input[i]]), 1);
        i += stride;
    }

    __syncthreads();

    if (tx < HISTOGRAM_LENGTH) {
        atomicAdd( &(histogram[tx]), private_histogram[tx]);
    }

    __syncthreads();
}
// END: Kernel to compute histogram

// BEGIN: Code to perform scan (from my MP 5.2 submission)
__global__ void scan(unsigned int *input, float *output, int len, int Norm) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from the host

    // Rename some relevant variables
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Load input into shared memory
    __shared__ float partial[2*BLOCK_SIZE_SCAN];

    int first_idx = tx + (bx * BLOCK_SIZE_SCAN) * 2;
    int second_idx = first_idx + BLOCK_SIZE_SCAN;

    if (first_idx < len) {
        partial[tx] = (float) input[first_idx] / Norm; // Must normalize to get CDF
        if (second_idx < len) {
            partial[tx + BLOCK_SIZE_SCAN] = (float) input[second_idx] / Norm; // Must normalize to get CDF
        } else {
            partial[tx + BLOCK_SIZE_SCAN] = 0.0f;
        }
    } else {
        partial[tx] = 0.0f;
        partial[tx + BLOCK_SIZE_SCAN] = 0.0f;
    }
    __syncthreads();

    // Scan step
    int stride = 1;
    while (stride < 2 * BLOCK_SIZE_SCAN) {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE_SCAN && (index - stride) >= 0) {
            partial[index] += partial[index-stride];
        }
        stride *= 2;
    }

    // Post scan step
    stride = BLOCK_SIZE_SCAN / 2;
    while (stride > 0) {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if ((index + stride) < 2 * BLOCK_SIZE_SCAN){
            partial[index + stride] += partial[index];
        }
        stride /= 2;
    }

    __syncthreads();

    // Write output
    if (first_idx < len) {
        output[first_idx] = partial[tx];
        if (second_idx < len) {
            output[second_idx] = partial[tx + BLOCK_SIZE_SCAN];
        }
    }

    __syncthreads();

}
// END: Code to perform scan (from my MP 5.2 submission)

// BEGIN: Kernel to apply histogram equalization function
__global__ void equalize(unsigned char *input, float *cdf, int Width, int Height, int Channel) {
    // Compute index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int w = bx * blockDim.x + tx; // Width
    int h = by * blockDim.y + ty; // Height
    int c = bz * blockDim.z + tz; // Channel; block index is unnecessary because there are 3 channels only


    if (w < Width && h < Height) {
        int pos = h * Width + w;
        int index = pos * Channel + c;
        unsigned char val = input[index];
        float cdfmin = cdf[0]; // Min must be first element, which can be zero
        float scaled = 255. * (cdf[val] - cdfmin) /(1. - cdfmin);
        float new_val = min(max(scaled, 0.), 255.);
        input[index] = (unsigned char) new_val;
    }
    __syncthreads();
}
// END: Kernel to apply histogram equalization function

// BEGIN: Kernel to cast image back to float
__global__ void cast_float(unsigned char *input, float *output, int Width, int Height, int Channel) {
    // Compute index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int w = bx * blockDim.x + tx; // Width
    int h = by * blockDim.y + ty; // Height
    int c = bz * blockDim.z + tz; // Channel; block index is unnecessary because there are 3 channels only


    if (w < Width && h < Height) {
        int pos = h * Width + w;
        int index = pos * Channel + c;
        output[index] = (float) input[index] / 255.;
    }
    __syncthreads();
}
// END: Kernel to cast image back to float

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
    // Declare device variables that require memory allocation
    float *deviceInputImage;
    unsigned char *deviceImageUnsChar;
    unsigned char *deviceImageGray;
    unsigned int *deviceHist;
    float *deviceCDF;
    float *deviceImageOutput;

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
    // Allocate device memory
    size_t size_InputImage = imageHeight * imageWidth * imageChannels * sizeof(float);
    size_t size_ImageUnsChar = imageHeight * imageWidth * imageChannels * sizeof(unsigned char);
    size_t size_ImageGray = imageHeight * imageWidth * sizeof(unsigned char);
    size_t size_Hist = HISTOGRAM_LENGTH * sizeof(unsigned int);
    size_t size_CDF = HISTOGRAM_LENGTH * sizeof(float);
    size_t size_ImageOutput = imageHeight * imageWidth * imageChannels * sizeof(float);

    wbCheck(cudaMalloc((void **)&deviceInputImage, size_InputImage));
    wbCheck(cudaMalloc((void **)&deviceImageUnsChar, size_ImageUnsChar));
    wbCheck(cudaMalloc((void **)&deviceImageGray, size_ImageGray));
    wbCheck(cudaMalloc((void **)&deviceHist, size_Hist));
    wbCheck(cudaMalloc((void **)&deviceCDF, size_CDF));
    wbCheck(cudaMalloc((void **)&deviceImageOutput, size_ImageOutput));

    //** DEBUGGING SECTION (YOU CAN IGNORE THIS) **//
    // Test cast as unsigned char (WORKS)
//    float test_input[65536 * 3] = {.1, .2, .3, .4, .5, 1.};
//    wbCheck(cudaMemcpy(deviceInputImage, test_input, size_InputImage, cudaMemcpyHostToDevice));
//
//    dim3 GridDim( ceil( (float) imageWidth / BLOCK_SIZE ), ceil( (float) imageHeight / BLOCK_SIZE ), 1);
//    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, imageChannels);
//
//    cast_unsigned_char<<<GridDim, BlockDim>>>(deviceInputImage, deviceImageUnsChar, imageWidth, imageHeight, imageChannels);
//    cudaDeviceSynchronize();
//    // Copy to host and print
//    unsigned char *hostImageUnsChar;
//    hostImageUnsChar = (unsigned char *)malloc(size_ImageUnsChar);
//    wbCheck(cudaMemcpy(hostImageUnsChar, deviceImageUnsChar, size_ImageUnsChar, cudaMemcpyDeviceToHost));
//
//    printf("Start uns char image:\n");
//    for (unsigned int i = 0; i < 20; i++) {
//        printf("%u  ", i);
//        printf("%hhu\n", hostImageUnsChar[i]);
//    }
//    printf("End uns char image\n");
//
//    for (unsigned int i = 6; i < 20; i++) {
//        if (hostImageUnsChar[i] != 0) {
//            printf("Index %u is not equal to zero.", i);
//        }
//    }

    // Test to grayscale (WORKS)
//    unsigned char test_input[65536 * 3] = {1, 1, 1, 255, 255, 255, 10, 10, 10};
//    wbCheck(cudaMemcpy(deviceImageUnsChar, test_input, size_ImageGray, cudaMemcpyHostToDevice));
//
//    dim3 GridDim2( ceil( (float) imageWidth / BLOCK_SIZE ), ceil( (float) imageHeight / BLOCK_SIZE ), 1);
//    dim3 BlockDim2(BLOCK_SIZE, BLOCK_SIZE, 1);
//    to_grayscale<<<GridDim2, BlockDim2>>>(deviceImageUnsChar, deviceImageGray, imageWidth, imageHeight, imageChannels);
//    cudaDeviceSynchronize();
//
//    // Copy to host and print
//    unsigned char *hostImageGray;
//    hostImageGray = (unsigned char *)malloc(size_ImageGray);
//    wbCheck(cudaMemcpy(hostImageGray, deviceImageGray, size_ImageGray, cudaMemcpyDeviceToHost));
//
//    printf("Start gray image:\n");
//    for (unsigned int i = 0; i < 20; i++) {
//        printf("%u  ", i);
//        printf("%hhu\n", hostImageGray[i]);
//    }
//    printf("End gray image\n");
//
//    for (unsigned int i = 3; i < imageWidth * imageHeight; i++) {
//        if (hostImageGray[i] != 0) {
//            printf("Index %u is not equal to zero.", i);
//        }
//    }

    // Test histogram code (WORKS)
//    unsigned int test_input[65536] = {1, 2, 3, 4, 5, 255};
//    wbCheck(cudaMemcpy(deviceImageGray, test_input, size_ImageGray, cudaMemcpyHostToDevice));
//
//    dim3 GridDim3( 1, 1, 1);
//    dim3 BlockDim3(HISTOGRAM_LENGTH, 1, 1);
//
//    compute_histogram<<<GridDim3, BlockDim3>>>(deviceImageGray, deviceHist, imageWidth * imageHeight);
//
//    // Copy to host and print
//    unsigned int *hostHist;
//    hostHist = (unsigned int *)malloc(size_Hist);
//    wbCheck(cudaMemcpy(hostHist, deviceHist, size_Hist, cudaMemcpyDeviceToHost));
//
//    printf("Start histogram:\n");
//    for (unsigned int i = 0; i < HISTOGRAM_LENGTH; i++) {
//        printf("%u  ", i);
//        printf("%u\n", hostHist[i]);
//    }
//    printf("End histogram\n");

    // Test scan kernel (WORKS)
//    unsigned int test_hist[256] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//    wbCheck(cudaMemcpy(deviceHist, test_hist, size_Hist, cudaMemcpyHostToDevice));
//
//    dim3 GridDim4(1, 1, 1);
//    dim3 BlockDim4(BLOCK_SIZE_SCAN, 1, 1);
//    scan<<<GridDim4, BlockDim4>>>(deviceHist, deviceCDF, HISTOGRAM_LENGTH, 45);
//    cudaDeviceSynchronize();

    // Copy to host and print
//    float *hostCDF;
//    hostCDF = (float *)malloc(size_CDF);
//    wbCheck(cudaMemcpy(hostCDF, deviceCDF, size_CDF, cudaMemcpyDeviceToHost));

//    printf("Start cdf:\n");
//    for (unsigned int i = 0; i < 20; i++) {
//        printf("%u  ", i);
//        printf("%f\n", hostCDF[i]);
//    }
//    printf("End cdf\n");

    // Test equalize kernel (WORKS)
//    unsigned char test_input[65536 * 3] = {1, 1, 1, 3, 3, 3, 2, 2, 2};
//    wbCheck(cudaMemcpy(deviceImageUnsChar, test_input, size_ImageUnsChar, cudaMemcpyHostToDevice));
//
//    dim3 GridDim( ceil( (float) imageWidth / BLOCK_SIZE ), ceil( (float) imageHeight / BLOCK_SIZE ), 1);
//    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, imageChannels);
//    equalize<<<GridDim, BlockDim>>>(deviceImageUnsChar, deviceCDF, imageWidth, imageHeight, imageChannels);
//    cudaDeviceSynchronize();
//
//    // Copy to host and print
//    unsigned char *hostImageUnsChar;
//    hostImageUnsChar = (unsigned char *)malloc(size_ImageUnsChar);
//    wbCheck(cudaMemcpy(hostImageUnsChar, deviceImageUnsChar, size_ImageUnsChar, cudaMemcpyDeviceToHost));
//
//    printf("Start uns char image (equalized):\n");
//    for (unsigned int i = 0; i < 20; i++) {
//        printf("%u  ", i);
//        printf("%hhu\n", hostImageUnsChar[i]);
//    }
//    printf("End uns char image (equalized)\n");
//
//    for (unsigned int i = 9; i < 65536 * 3; i++) {
//        if (hostImageUnsChar[i] != 0) {
//            printf("Index %u is not equal to zero.\n", i);
//        }
//    }
//
//    // Test cast to float (WORKS)
//    cast_float<<<GridDim, BlockDim>>>(deviceImageUnsChar, deviceImageOutput, imageWidth, imageHeight, imageChannels);
//    cudaDeviceSynchronize();
//
//    // Copy to host and print
//    wbCheck(cudaMemcpy(hostOutputImageData, deviceImageOutput, size_ImageOutput,cudaMemcpyDeviceToHost));
//
//    printf("Start output image:\n");
//    for (unsigned int i = 0; i < 20; i++) {
//        printf("%u  ", i);
//        printf("%f\n", hostOutputImageData[i]);
//    }
//    printf("End output image\n");
//
//    for (unsigned int i = 9; i < 65536 * 3; i++) {
//        if (hostOutputImageData[i] > 0.0000001) {
//            printf("Index %u is greater zero.\n", i);
//        }
//    }
    //** END DEBUGGING SECTION **//

    // Copy host memory to device memory
    wbCheck(cudaMemcpy(deviceInputImage, hostInputImageData, size_InputImage,cudaMemcpyHostToDevice));

    // Cast image as unsigned char
    dim3 GridDim( ceil( (float) imageWidth / BLOCK_SIZE ), ceil( (float) imageHeight / BLOCK_SIZE ), 1);
    dim3 BlockDim(BLOCK_SIZE, BLOCK_SIZE, imageChannels);

    cast_unsigned_char<<<GridDim, BlockDim>>>(deviceInputImage, deviceImageUnsChar, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    // Convert image to grayscale
    dim3 GridDim2( ceil( (float) imageWidth / BLOCK_SIZE ), ceil( (float) imageHeight / BLOCK_SIZE ), 1);
    dim3 BlockDim2(BLOCK_SIZE, BLOCK_SIZE, 1);

    to_grayscale<<<GridDim2, BlockDim2>>>(deviceImageUnsChar, deviceImageGray, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    // Compute histogram
    dim3 GridDim3( 1, 1, 1);
    dim3 BlockDim3(HISTOGRAM_LENGTH, 1, 1);

    compute_histogram<<<GridDim3, BlockDim3>>>(deviceImageGray, deviceHist, imageWidth * imageHeight);
    cudaDeviceSynchronize();

    // Compute CDF
    dim3 GridDim4(1, 1, 1);
    dim3 BlockDim4(BLOCK_SIZE_SCAN, 1, 1);

    scan<<<GridDim4, BlockDim4>>>(deviceHist, deviceCDF, HISTOGRAM_LENGTH, imageWidth * imageHeight);
    cudaDeviceSynchronize();

    // Apply equalization
    equalize<<<GridDim, BlockDim>>>(deviceImageUnsChar, deviceCDF, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    // Cast as float
    cast_float<<<GridDim, BlockDim>>>(deviceImageUnsChar, deviceImageOutput, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    // Copy output back to host
    wbCheck(cudaMemcpy(hostOutputImageData, deviceImageOutput, size_ImageOutput,cudaMemcpyDeviceToHost));

    wbSolution(args, outputImage);

    //@@ insert code here
    wbCheck(cudaFree(deviceInputImage));
    wbCheck(cudaFree(deviceImageUnsChar));
    wbCheck(cudaFree(deviceImageGray));
    wbCheck(cudaFree(deviceHist));
    wbCheck(cudaFree(deviceCDF));
    wbCheck(cudaFree(deviceImageOutput));

    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}
