// I HAVE INCLUDED ALL ITERATIONS OF MY CODE IN THIS FILE
// THEY ARE LABELLED "OPTIMIZATION <#>" FOLLOWING THE SAME ORDER AS MY REPORT
// THE ONLY UNCOMMENTED IMPLEMENTATION IS THE LAST ONE, OPTIMIZATION 6 (WHICH IS THE FASTEST ONE AND ALSO INCLUDES THE OPTIMIZATIONS IMPLEMENTED IN 1, 2, AND 3)

// THE LIST BELOW SHOWS WHAT OPTIMIZATIONS ARE IMPLEMENTED IN EACH ITERATION
// OPTIMIZATION 1
// Weight matrix (kernel values) in constant memory (1 point)

// OPTIMIZATION 2
// Multiple kernel implementations for different layer sizes (1 point)

// OPTIMIZATION 3
// Weight matrix (kernel values) in constant memory (1 point)
// Multiple kernel implementations for different layer sizes (1 point)
// Tiled shared memory convolution (2 points)

// OPTIMIZATION 4
// Weight matrix (kernel values) in constant memory (1 point)
// Multiple kernel implementations for different layer sizes (1 point)
// Shared memory matrix multiplication and input matrix unrolling (3 points)

// OPTIMIZATION 5
// Weight matrix (kernel values) in constant memory (1 point)
// Multiple kernel implementations for different layer sizes (1 point)
// Shared memory matrix multiplication and input matrix unrolling (3 points)
// Kernel fusion for unrolling and matrix-multiplication (requires previous optimization) (2 points)

// OPTIMIZATION 6
// Weight matrix (kernel values) in constant memory (1 point)
// Multiple kernel implementations for different layer sizes (1 point)
// Tiled shared memory convolution (2 points)
// Input channel reduction: atomics (2 point)

// TOTAL POINTS ATTEMPTED: 11 POINTS


// START: OPTIMIZATION 1 //
//#include <cmath>
//#include <iostream>
//#include "gpu-new-forward.h"
//
//#define TILE_WIDTH 16
//#define MASK_WIDTH 7
//__constant__ float cached_mask[4096];
//
//__global__ void conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//
//{
//    /*
//    Modify this function to implement the forward pass described in Chapter 16.
//    We have added an additional dimension to the tensors to support an entire mini-batch
//    The goal here is to be correct AND fast.
//
//    Function paramter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
//
//    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//    // An example use of these macros:
//    // float a = in_4d(0,0,0,0)
//    // out_4d(0,0,0,0) = a
//
//    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//    #define mask_4d(i3, i2, i1, i0) cached_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//    // Insert your GPU convolution kernel code here
//    int m = blockIdx.x; // Output map feature index
//    const int W_size = ceil((float) Width_out / TILE_WIDTH); // Number of horizontal tiles
//    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y; // Row
//    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x; // Column
//    int b = blockIdx.z; // Batch element index
//
//    float acc = 0.0f; // Initialize sum
//
//    if (h < Height_out && w < Width_out) {
//
//        for (int c = 0; c < Channel; c++) { // Sum across input channels
//            for (int p = 0; p < K; p++) { // Loop over filter dimensions
//                for (int q = 0; q < K; q++) {
//                    acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
//                }
//            }
//        }
//
//        out_4d(b, m, h, w) = acc;
//    }
//
//    #undef out_4d
//    #undef in_4d
//    #undef mask_4d
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Allocate memory and copy over the relevant data structures to the GPU
//
//    // We pass double pointers for you to initialize the relevant device pointers,
//    //  which are passed to the other two functions.
//
//    // Useful snippet for error checking
//    // cudaError_t error = cudaGetLastError();
//    // if(error != cudaSuccess)
//    // {
//    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//    //     exit(-1);
//    // }
//
//    // Allocate GPU memory for input array
//    size_t sizeInput = Batch * Channel * Height * Width * sizeof(float);
//    cudaMalloc((void **) device_input_ptr, sizeInput);
//
//    // Calculate and allocate GPU memory needed for output array
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//    cudaMalloc((void **) device_output_ptr, sizeOutput);
//
//    // Allocate memory for mask filter (might need to change for constant memory)
//    size_t sizeMask = Map_out * Channel * K * K * sizeof(float);
////    cudaMalloc((void **) device_mask_ptr, sizeMask);
//
//    // Transfer the input and mask to GPU
//    cudaMemcpy(*device_input_ptr, host_input, sizeInput, cudaMemcpyHostToDevice);
////    cudaMemcpy(*device_mask_ptr, host_mask, sizeMask, cudaMemcpyHostToDevice);
//    cudaMemcpyToSymbol(cached_mask, host_mask, sizeMask);
//
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Set the kernel dimensions and call the kernel
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//
//    const int GridX = Map_out;
//    const int GridY = ceil((float) Height_out/TILE_WIDTH) * ceil((float) Width_out/TILE_WIDTH);
//    const int GridZ = Batch;
//    dim3 DimGrid(GridX, GridY, GridZ);
//    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//
////    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);
//    cudaDeviceSynchronize();
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Copy the output back to host
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//
//    cudaMemcpy(host_output, device_output, sizeOutput, cudaMemcpyDeviceToHost);
//
//    // Free device memory
//    cudaFree(device_input);
//    cudaFree(device_output);
////    cudaFree(device_mask);
//}
//
//
//__host__ void GPUInterface::get_device_properties()
//{
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    for(int dev = 0; dev < deviceCount; dev++)
//    {
//        cudaDeviceProp deviceProp;
//        cudaGetDeviceProperties(&deviceProp, dev);
//
//        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//    }
//}
// END: OPTIMIZATION 1 //

// START: OPTIMIZATION 2 //
//#include <cmath>
//#include <iostream>
//#include "gpu-new-forward.h"
//
//#define TILE_WIDTH_INPUT 16
//#define TILE_WIDTH_HIDDEN 17
//
//__global__ void conv_forward_kernel_input_layer(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
///*
//    Modify this function to implement the forward pass described in Chapter 16.
//    We have added an additional dimension to the tensors to support an entire mini-batch
//    The goal here is to be correct AND fast.
//    Function paramter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
//
//    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//    // An example use of these macros:
//    // float a = in_4d(0,0,0,0)
//    // out_4d(0,0,0,0) = a
//
//#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//    // Insert your GPU convolution kernel code here
//    int m = blockIdx.x; // Output map feature index
//    const int W_size = Width_out / TILE_WIDTH_INPUT; // Number of horizontal tiles
//    int h = (blockIdx.y / W_size) * TILE_WIDTH_INPUT + threadIdx.y; // Row
//    int w = (blockIdx.y % W_size) * TILE_WIDTH_INPUT + threadIdx.x; // Column
//    int b = blockIdx.z; // Batch element index
//
//    float acc = 0.0f; // Initialize sum
//
//    for (int c = 0; c < Channel; c++) { // Sum across input channels
//        for (int p = 0; p < K; p++) { // Loop over filter dimensions
//            for (int q = 0; q < K; q++) {
//                acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
//            }
//        }
//    }
//
//    out_4d(b, m, h, w) = acc;
//
//#undef out_4d
//#undef in_4d
//#undef mask_4d
//}
//
//__global__ void conv_forward_kernel_hidden_layer(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    /*
//    Modify this function to implement the forward pass described in Chapter 16.
//    We have added an additional dimension to the tensors to support an entire mini-batch
//    The goal here is to be correct AND fast.
//    Function paramter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
//
//    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//    // An example use of these macros:
//    // float a = in_4d(0,0,0,0)
//    // out_4d(0,0,0,0) = a
//
//#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//    // Insert your GPU convolution kernel code here
//    int m = blockIdx.x; // Output map feature index
//    const int W_size = Width_out / TILE_WIDTH_HIDDEN; // Number of horizontal tiles
//    int h = (blockIdx.y / W_size) * TILE_WIDTH_HIDDEN + threadIdx.y; // Row
//    int w = (blockIdx.y % W_size) * TILE_WIDTH_HIDDEN + threadIdx.x; // Column
//    int b = blockIdx.z; // Batch element index
//
//    float acc = 0.0f; // Initialize sum
//
//    for (int c = 0; c < Channel; c++) { // Sum across input channels
//        for (int p = 0; p < K; p++) { // Loop over filter dimensions
//            for (int q = 0; q < K; q++) {
//                acc += in_4d(b, c, h + p, w + q) * mask_4d(m, c, p, q);
//            }
//        }
//    }
//
//    out_4d(b, m, h, w) = acc;
//
//#undef out_4d
//#undef in_4d
//#undef mask_4d
//}
//
//__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Allocate memory and copy over the relevant data structures to the GPU
//
//    // We pass double pointers for you to initialize the relevant device pointers,
//    //  which are passed to the other two functions.
//
//    // Useful snippet for error checking
//    // cudaError_t error = cudaGetLastError();
//    // if(error != cudaSuccess)
//    // {
//    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//    //     exit(-1);
//    // }
//
//    // Allocate GPU memory for input array
//    size_t sizeInput = Batch * Channel * Height * Width * sizeof(float);
//    cudaMalloc((void **) device_input_ptr, sizeInput);
//
//    // Calculate and allocate GPU memory needed for output array
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//    cudaMalloc((void **) device_output_ptr, sizeOutput);
//
//    // Allocate memory for mask filter (might need to change for constant memory)
//    size_t sizeMask = Map_out * Channel * K * K * sizeof(float);
//    cudaMalloc((void **) device_mask_ptr, sizeMask);
//
//    // Transfer the input and mask to GPU
//    cudaMemcpy(*device_input_ptr, host_input, sizeInput, cudaMemcpyHostToDevice);
//    cudaMemcpy(*device_mask_ptr, host_mask, sizeMask, cudaMemcpyHostToDevice);
//
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Set the kernel dimensions and call the kernel
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//
//    const int GridX = Map_out;
//    const int GridZ = Batch;
//
//    if (Height_out == 80 && Width_out == 80) {
//        // Use kernel for input layer
//        const int GridY = (Height_out/TILE_WIDTH_INPUT) * (Width_out/TILE_WIDTH_INPUT);
//
//        dim3 DimGrid(GridX, GridY, GridZ);
//        dim3 DimBlock(TILE_WIDTH_INPUT, TILE_WIDTH_INPUT, 1);
//
//        conv_forward_kernel_input_layer<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//        cudaDeviceSynchronize();
//
//    } else if (Height_out == 34 && Width_out == 34) {
//        // Use kernel for hidden layer
//        const int GridY = (Height_out/TILE_WIDTH_HIDDEN) * (Width_out/TILE_WIDTH_HIDDEN);
//
//        dim3 DimGrid(GridX, GridY, GridZ);
//        dim3 DimBlock(TILE_WIDTH_HIDDEN, TILE_WIDTH_HIDDEN, 1);
//
//        conv_forward_kernel_hidden_layer<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//        cudaDeviceSynchronize();
//
//    } else {
//        printf("I fell into an exception!\n");
//    }
//
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Copy the output back to host
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//
//    cudaMemcpy(host_output, device_output, sizeOutput, cudaMemcpyDeviceToHost);
//
//    // Free device memory
//    cudaFree(device_input);
//    cudaFree(device_output);
//    cudaFree(device_mask);
//}
//
//
//__host__ void GPUInterface::get_device_properties()
//{
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    for(int dev = 0; dev < deviceCount; dev++)
//    {
//        cudaDeviceProp deviceProp;
//        cudaGetDeviceProperties(&deviceProp, dev);
//
//        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//    }
//}
// END: OPTIMIZATION 2 //

// START: OPTIMIZATION 3 //
//#include <cmath>
//#include <iostream>
//#include "gpu-new-forward.h"
//
//#define TILE_WIDTH_INPUT 16
//#define TILE_WIDTH_HIDDEN 17
//#define MASK_WIDTH 7
//__constant__ float cached_mask[3136];
//
//__global__ void conv_forward_kernel_input_layer(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
///*
//    Modify this function to implement the forward pass described in Chapter 16.
//    We have added an additional dimension to the tensors to support an entire mini-batch
//    The goal here is to be correct AND fast.
//    Function paramter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define mask_4d(i3, i2, i1, i0) cached_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//    // Insert your GPU convolution kernel code here
//    int m = blockIdx.x; // Output map feature index
//    const int W_size = Width_out / TILE_WIDTH_INPUT; // Number of horizontal tiles
//    int first_output_row = (blockIdx.y / W_size) * TILE_WIDTH_INPUT;
//    int first_output_col = (blockIdx.y % W_size) * TILE_WIDTH_INPUT;
//    int h = first_output_row + threadIdx.y; // Row
//    int w = first_output_col + threadIdx.x; // Column
//    int b = blockIdx.z; // Batch element index
//
//
//    // Load tile into shared memory
//    const int input_tile_size = TILE_WIDTH_INPUT + K - 1; // = 22
//    __shared__ float tile[1][22][22];
//    int load_iters = ceil( (float)  (input_tile_size*input_tile_size*Channel) / (TILE_WIDTH_INPUT*TILE_WIDTH_INPUT));
//    int one_dim_index_out, c, k, l;
//    for (int i = 0; i < load_iters; i++){
//        one_dim_index_out = threadIdx.y * TILE_WIDTH_INPUT + threadIdx.x + (i * TILE_WIDTH_INPUT * TILE_WIDTH_INPUT); // One dimensional index for output tile element
//        c = one_dim_index_out / (input_tile_size * input_tile_size); // Input channel
//        one_dim_index_out = one_dim_index_out - c * input_tile_size * input_tile_size; // Lower dimensionality of index
//        k = one_dim_index_out / input_tile_size; // Input row
//        l = one_dim_index_out % input_tile_size; // Input col
//        if (c < Channel && k < input_tile_size && l < input_tile_size){
//            tile[c][k][l] = in_4d(b, c, first_output_row + k, first_output_col + l);
//        }
//    }
//    __syncthreads();
//
//    float acc = 0.0f; // Initialize sum
//
//    for (int c = 0; c < Channel; c++) { // Sum across input channels
//        for (int p = 0; p < K; p++) { // Loop over filter dimensions
//            for (int q = 0; q < K; q++) {
//                acc += tile[c][threadIdx.y + p][threadIdx.x + q] * mask_4d(m, c, p, q);
//            }
//        }
//    }
//
//    out_4d(b, m, h, w) = acc;
//
//#undef out_4d
//#undef in_4d
//#undef mask_4d
//}
//
//__global__ void conv_forward_kernel_hidden_layer(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    /*
//    Modify this function to implement the forward pass described in Chapter 16.
//    We have added an additional dimension to the tensors to support an entire mini-batch
//    The goal here is to be correct AND fast.
//    Function paramter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define mask_4d(i3, i2, i1, i0) cached_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//    // Insert your GPU convolution kernel code here
//    int m = blockIdx.x; // Output map feature index
//    const int W_size = Width_out / TILE_WIDTH_HIDDEN; // Number of horizontal tiles
//    int first_output_row = (blockIdx.y / W_size) * TILE_WIDTH_HIDDEN;
//    int first_output_col = (blockIdx.y % W_size) * TILE_WIDTH_HIDDEN;
//    int h = first_output_row + threadIdx.y; // Row
//    int w = first_output_col + threadIdx.x; // Column
//    int b = blockIdx.z; // Batch element index
//
//    // Load tile into shared memory
//    const int input_tile_size = TILE_WIDTH_HIDDEN + K - 1;
//    __shared__ float tile[4][23][23];
//    int load_iters = ceil( (float)  (input_tile_size*input_tile_size*Channel) / (TILE_WIDTH_HIDDEN*TILE_WIDTH_HIDDEN));
//    int one_dim_index_out, c, k, l;
//    for (int i = 0; i < load_iters; i++){
//        one_dim_index_out = threadIdx.y * TILE_WIDTH_HIDDEN + threadIdx.x + (i * TILE_WIDTH_HIDDEN * TILE_WIDTH_HIDDEN); // One dimensional index for output tile element
//        c = one_dim_index_out / (input_tile_size * input_tile_size); // Input channel
//        one_dim_index_out = one_dim_index_out - c * input_tile_size * input_tile_size; // Lower dimensionality of index
//        k = one_dim_index_out / input_tile_size; // Input row
//        l = one_dim_index_out % input_tile_size; // Input col
//        if (c < Channel && k < input_tile_size && l < input_tile_size){
//            tile[c][k][l] = in_4d(b, c, first_output_row + k, first_output_col + l);
//        }
//    }
//    __syncthreads();
//
//    float acc = 0.0f; // Initialize sum
//
//    for (int c = 0; c < Channel; c++) { // Sum across input channels
//        for (int p = 0; p < K; p++) { // Loop over filter dimensions
//            for (int q = 0; q < K; q++) {
//                acc += tile[c][threadIdx.y + p][threadIdx.x + q] * mask_4d(m, c, p, q);
//            }
//        }
//    }
//
//    out_4d(b, m, h, w) = acc;
//
//#undef out_4d
//#undef in_4d
//#undef mask_4d
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Allocate memory and copy over the relevant data structures to the GPU
//
//    // We pass double pointers for you to initialize the relevant device pointers,
//    //  which are passed to the other two functions.
//
//    // Useful snippet for error checking
//    // cudaError_t error = cudaGetLastError();
//    // if(error != cudaSuccess)
//    // {
//    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//    //     exit(-1);
//    // }
//
//    // Allocate GPU memory for input array
//    size_t sizeInput = Batch * Channel * Height * Width * sizeof(float);
//    cudaMalloc((void **) device_input_ptr, sizeInput);
//
//    // Calculate and allocate GPU memory needed for output array
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//    cudaMalloc((void **) device_output_ptr, sizeOutput);
//
//    // Allocate memory for mask filter (might need to change for constant memory)
//    size_t sizeMask = Map_out * Channel * K * K * sizeof(float);
//
//    // Transfer the input and mask to GPU
//    cudaMemcpy(*device_input_ptr, host_input, sizeInput, cudaMemcpyHostToDevice);
//    cudaMemcpyToSymbol(cached_mask, host_mask, sizeMask);
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Set the kernel dimensions and call the kernel
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//    const int GridX = Map_out;
//    const int GridZ = Batch;
//
//    if (Height_out == 80 && Width_out == 80) {
//        // Use kernel for input layer
//        const int GridY = (Height_out/TILE_WIDTH_INPUT) * (Width_out/TILE_WIDTH_INPUT);
//
//        dim3 DimGrid(GridX, GridY, GridZ);
//        dim3 DimBlock(TILE_WIDTH_INPUT, TILE_WIDTH_INPUT, 1);
//
//        conv_forward_kernel_input_layer<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//        cudaDeviceSynchronize();
//
//    } else if (Height_out == 34 && Width_out == 34) {
//        // Use kernel for hidden layer
//        const int GridY = (Height_out/TILE_WIDTH_HIDDEN) * (Width_out/TILE_WIDTH_HIDDEN);
//
//        dim3 DimGrid(GridX, GridY, GridZ);
//        dim3 DimBlock(TILE_WIDTH_HIDDEN, TILE_WIDTH_HIDDEN, 1);
//
//        conv_forward_kernel_hidden_layer<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//        cudaDeviceSynchronize();
//
//    } else {
//        printf("I fell into an exception!\n");
//    }
//
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Copy the output back to host
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//
//    cudaMemcpy(host_output, device_output, sizeOutput, cudaMemcpyDeviceToHost);
//
//    // Free device memory
//    cudaFree(device_input);
//    cudaFree(device_output);
//}
//
//
//__host__ void GPUInterface::get_device_properties()
//{
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    for(int dev = 0; dev < deviceCount; dev++)
//    {
//        cudaDeviceProp deviceProp;
//        cudaGetDeviceProperties(&deviceProp, dev);
//
//        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//    }
//}
// END: OPTIMIZATION 3 //

// START: OPTIMIZATION 4 //
//#include <cmath>
//#include <iostream>
//#include "gpu-new-forward.h"
//
//#define TILE_WIDTH_INPUT 16
//#define TILE_WIDTH_HIDDEN 17
//#define MASK_WIDTH 7
//#define TILE_WIDTH_MATRIX_INPUT 16
//#define TILE_HEIGHT_MATRIX_INPUT 49
//#define TILE_WIDTH_MATRIX_HIDDEN 17
//#define TILE_HEIGHT_MATRIX_HIDDEN 49
//__constant__ float cached_mask[3136];
//
//// Unrolls a batch of input matrices
//__global__ void matrix_unroll_input(const float *input, float *unrolled_matrix, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int Batch_offset)
//{
//    /*
//    Store the input as an unrolled matrix
//    Function parameter definitions:
//    input - input
//    unrolled_matrix - memory to set as the unrolled matrix
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    const int Height_out_unrolled = (Channel) * (K * K);
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define unrolled_3d(i2, i1, i0)  unrolled_matrix[(i2) * (Height_out_unrolled * Width_out_unrolled) + (i1) * (Width_out_unrolled) + (i0)]
//
//    const int W_size = Width_out / TILE_WIDTH_INPUT; // Number of horizontal tiles
//    int first_output_row = (blockIdx.y / W_size) * TILE_WIDTH_INPUT;
//    int first_output_col = (blockIdx.y % W_size) * TILE_WIDTH_INPUT;
//    int h = first_output_row + threadIdx.y; // Row
//    int w = first_output_col + threadIdx.x; // Column
//    int b = blockIdx.z; // Batch element index
//
//    for (int c = 0; c < Channel; c++) {
//        int w_base = c * (K*K);
//        for (int p = 0; p < K; p++) { // Loop over filter dimensions
//            for (int q = 0; q < K; q++) {
//                int h_unroll = w_base + p * K + q;
//                int w_unroll = h * Width_out + w;
//                unrolled_3d(b, h_unroll, w_unroll) = in_4d(b + Batch_offset, c, h + p, w + q);
//            }
//        }
//    }
//
//#undef in_4d
//#undef unrolled_3d
//}
//
//__global__ void matrix_unroll_hidden(const float *input, float *unrolled_matrix, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int Batch_offset)
//{
//    /*
//    Store the input as an unrolled matrix
//    Function parameter definitions:
//    input - input
//    unrolled_matrix - memory to set as the unrolled matrix
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    const int Height_out_unrolled = (Channel) * (K * K);
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define unrolled_3d(i2, i1, i0)  unrolled_matrix[(i2) * (Height_out_unrolled * Width_out_unrolled) + (i1) * (Width_out_unrolled) + (i0)]
//
//    const int W_size = Width_out / TILE_WIDTH_HIDDEN; // Number of horizontal tiles
//    int first_output_row = (blockIdx.y / W_size) * TILE_WIDTH_HIDDEN;
//    int first_output_col = (blockIdx.y % W_size) * TILE_WIDTH_HIDDEN;
//    int h = first_output_row + threadIdx.y; // Row
//    int w = first_output_col + threadIdx.x; // Column
//    int b = blockIdx.z; // Batch element index
//
//    for (int c = 0; c < Channel; c++) {
//        int w_base = c * (K*K);
//        for (int p = 0; p < K; p++) { // Loop over filter dimensions
//            for (int q = 0; q < K; q++) {
//                int h_unroll = w_base + p * K + q;
//                int w_unroll = h * Width_out + w;
//                unrolled_3d(b, h_unroll, w_unroll) = in_4d(b + Batch_offset, c, h + p, w + q);
//            }
//        }
//    }
//
//#undef in_4d
//#undef unrolled_3d
//}
//
//// Performs convolution with matrix multiplication
//__global__ void conv_forward_kernel_input_layer_matrix_unroll(float *output, const float *unrolled_matrix, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int Batch_offset)
//{
//    /*
//    Function parameter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    // Compute output dimensions
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    const int Height_out_unrolled = (Channel) * (K * K);
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//
//    // Rename some useful variables
//    const int bx = blockIdx.x;
//    const int by = blockIdx.y;
//    const int bz = blockIdx.z;
//    const int tx = threadIdx.x;
//    const int ty = threadIdx.y;
//
//#define out_3d(i2, i1, i0) output[(i2) * (Map_out * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
//#define unrolled_3d(i2, i1, i0)  unrolled_matrix[(i2) * (Height_out_unrolled * Width_out_unrolled) + (i1) * (Width_out_unrolled) + (i0)]
//#define mask_2d(i1, i0) cached_mask[(i1) * (Channel * K * K) + (i0)]
//
//    int w = bx * blockDim.x + tx; // Col
//    int h = by * blockDim.y + ty; // Row
//    int b = bz;
//
//    __shared__ float tile[TILE_HEIGHT_MATRIX_INPUT][TILE_WIDTH_MATRIX_INPUT];
//
//    float acc = 0.;
//    // Load shared memory tile
//    tile[ty][tx] = unrolled_3d(b, ty, w);
//    __syncthreads();
//
//    // Compute sum
//    for (int p = 0; p < TILE_HEIGHT_MATRIX_INPUT; p++) {
//        acc += tile[p][tx] * mask_2d(h, p);
//    }
//    __syncthreads();
//
//    // Save sum to output
//    if (h < Map_out && w < Width_out_unrolled) {
//        out_3d(b + Batch_offset, h, w) = acc;
//    }
//
//#undef out_3d
//#undef unrolled_3d
//#undef mask_2d
//}
//
//__global__ void conv_forward_kernel_hidden_layer_matrix_unroll(float *output, const float *unrolled_matrix, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int Batch_offset)
//{
//    /*
//    Function parameter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    // Compute output dimensions
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    const int Height_out_unrolled = (Channel) * (K * K);
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//
//    // Rename some useful variables
//    const int bx = blockIdx.x;
//    const int by = blockIdx.y;
//    const int bz = blockIdx.z;
//    const int tx = threadIdx.x;
//    const int ty = threadIdx.y;
//
//#define out_3d(i2, i1, i0) output[(i2) * (Map_out * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
//#define unrolled_3d(i2, i1, i0)  unrolled_matrix[(i2) * (Height_out_unrolled * Width_out_unrolled) + (i1) * (Width_out_unrolled) + (i0)]
//#define mask_2d(i1, i0) cached_mask[(i1) * (Channel * K * K) + (i0)]
//
//    int w = bx * blockDim.x + tx; // Col
//    int h = by * blockDim.y + ty; // Row
//    int b = bz;
//
//    __shared__ float tile[TILE_HEIGHT_MATRIX_HIDDEN][TILE_WIDTH_MATRIX_HIDDEN];
//
//    const int iters = Height_out_unrolled / TILE_HEIGHT_MATRIX_HIDDEN; // Number of iterations to compute one element
//    int height_offset;
//    float acc = 0.;
//    for (int i = 0; i < iters; i++) {
//        // Load shared memory tile
//        height_offset = i * TILE_HEIGHT_MATRIX_HIDDEN;
//        tile[ty][tx] = unrolled_3d(b, ty + height_offset, w);
//        __syncthreads();
//
//        // Compute partial sum
//        for (int p = 0; p < TILE_HEIGHT_MATRIX_HIDDEN; p++) {
//            acc += tile[p][tx] * mask_2d(h, p + height_offset);
//        }
//        __syncthreads();
//    }
//
//    // Save sum to output
//    if (h < Map_out && w < Width_out_unrolled) {
//        out_3d(b + Batch_offset, h, w) = acc;
//    }
//
//#undef out_3d
//#undef unrolled_3d
//#undef mask_2d
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Allocate memory and copy over the relevant data structures to the GPU
//
//    // We pass double pointers for you to initialize the relevant device pointers,
//    //  which are passed to the other two functions.
//
//    // Useful snippet for error checking
//    // cudaError_t error = cudaGetLastError();
//    // if(error != cudaSuccess)
//    // {
//    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//    //     exit(-1);
//    // }
//
//    // Allocate GPU memory for input array
//    size_t sizeInput = Batch * Channel * Height * Width * sizeof(float);
//    cudaMalloc((void **) device_input_ptr, sizeInput);
//
//    // Calculate and allocate GPU memory needed for output array
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//    cudaMalloc((void **) device_output_ptr, sizeOutput);
//
//    // Allocate memory for mask filter (might need to change for constant memory)
//    size_t sizeMask = Map_out * Channel * K * K * sizeof(float);
//
//    // Transfer the input and mask to GPU
//
//    cudaMemcpy(*device_input_ptr, host_input, sizeInput, cudaMemcpyHostToDevice);
//    cudaMemcpyToSymbol(cached_mask, host_mask, sizeMask);
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Set iteration loop over batch
//    int iters, samples_per_batch;
//    if (Batch == 10000) {
//        iters = 10;
//        samples_per_batch = Batch / iters;
//    } else {
//        iters = 1;
//        samples_per_batch = Batch;
//    }
//
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//
//    // Unrolled dimensions and memory allocation
//    const int Height_out_unrolled = (Channel) * (K * K);
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//    size_t sizeUnrolled = samples_per_batch * Height_out_unrolled * Width_out_unrolled * sizeof(float);
//
//    float *device_unrolled;
//    cudaMalloc((void **) &device_unrolled, sizeUnrolled);
//
//    const int GridZ = samples_per_batch;
//
//    for (int iter = 0; iter < iters; iter++) {
//
//        int batch_offset = iter * samples_per_batch;
//
//        if (Height_out == 80 && Width_out == 80) {
//            // Use kernels for input layer
//            const int GridY = (Height_out / TILE_WIDTH_INPUT) * (Width_out / TILE_WIDTH_INPUT);
//
//            dim3 DimGrid(1, GridY, GridZ);
//            dim3 DimBlock(TILE_WIDTH_INPUT, TILE_WIDTH_INPUT, 1);
//
//            matrix_unroll_input<<<DimGrid, DimBlock>>>(device_input, device_unrolled, Batch, Map_out, Channel, Height, Width, K, batch_offset);
//            cudaDeviceSynchronize();
//
//            const int GridX2 = (Width_out_unrolled / TILE_WIDTH_MATRIX_INPUT);
//            const int GridY2 = (Height_out_unrolled / TILE_HEIGHT_MATRIX_INPUT);
//
//            dim3 DimGrid2(GridX2, GridY2, GridZ);
//            dim3 DimBlock2(TILE_WIDTH_MATRIX_INPUT, TILE_HEIGHT_MATRIX_INPUT, 1);
//
//            conv_forward_kernel_input_layer_matrix_unroll<<<DimGrid2, DimBlock2>>>(device_output, device_unrolled, device_mask, Batch, Map_out, Channel, Height, Width, K, batch_offset);
//            cudaDeviceSynchronize();
//
//        } else if (Height_out == 34 && Width_out == 34) {
//            // Use kernels for hidden layer
//            const int GridY = (Height_out / TILE_WIDTH_HIDDEN) * (Width_out / TILE_WIDTH_HIDDEN);
//
//            dim3 DimGrid(1, GridY, GridZ);
//            dim3 DimBlock(TILE_WIDTH_HIDDEN, TILE_WIDTH_HIDDEN, 1);
//
//            matrix_unroll_hidden<<<DimGrid, DimBlock>>>(device_input, device_unrolled, Batch, Map_out, Channel, Height, Width, K, batch_offset);
//            cudaDeviceSynchronize();
//
//            const int GridX2 = (Width_out_unrolled / TILE_WIDTH_MATRIX_HIDDEN);
//            const int GridY2 = (Height_out_unrolled / TILE_HEIGHT_MATRIX_HIDDEN);
//
//            dim3 DimGrid2(GridX2, GridY2, GridZ);
//            dim3 DimBlock2(TILE_WIDTH_MATRIX_HIDDEN, TILE_HEIGHT_MATRIX_HIDDEN, 1);
//
//            conv_forward_kernel_hidden_layer_matrix_unroll<<<DimGrid2, DimBlock2>>>(device_output, device_unrolled, device_mask, Batch, Map_out, Channel, Height, Width, K, batch_offset);
//            cudaDeviceSynchronize();
//
//        } else {
//            printf("I fell into an exception!\n");
//        }
//    }
//
//    cudaFree(device_unrolled);
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Copy the output back to host
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//
//    cudaMemcpy(host_output, device_output, sizeOutput, cudaMemcpyDeviceToHost);
//
//    // Free device memory
//    cudaFree(device_input);
//    cudaFree(device_output);
//}
//
//
//__host__ void GPUInterface::get_device_properties()
//{
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    for(int dev = 0; dev < deviceCount; dev++)
//    {
//        cudaDeviceProp deviceProp;
//        cudaGetDeviceProperties(&deviceProp, dev);
//
//        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//    }
//}
// END: OPTIMIZATION 4 //

// START: OPTIMIZATION 5 //
//#include <cmath>
//#include <iostream>
//#include "gpu-new-forward.h"
//
//#define TILE_WIDTH_MATRIX_INPUT 16
//#define TILE_HEIGHT_MATRIX_INPUT 49
//#define TILE_WIDTH_MATRIX_HIDDEN 17
//#define TILE_HEIGHT_MATRIX_HIDDEN 49
//__constant__ float cached_mask[3136];
//
//// Performs convolution with matrix multiplication
//__global__ void conv_forward_kernel_input_layer_matrix_unroll(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    /*
//    Function parameter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    // Compute output dimensions
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//
//    // Rename some useful variables
//    const int bx = blockIdx.x;
//    const int by = blockIdx.y;
//    const int bz = blockIdx.z;
//    const int tx = threadIdx.x;
//    const int ty = threadIdx.y;
//
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define out_3d(i2, i1, i0) output[(i2) * (Map_out * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
//#define mask_2d(i1, i0) cached_mask[(i1) * (Channel * K * K) + (i0)]
//
//    int w = bx * blockDim.x + tx; // Col of unrolled matrix
//    int h = by * blockDim.y + ty; // Row of unrolled matrix --> Col of mask
//    int b = bz;
//
//    __shared__ float tile[TILE_HEIGHT_MATRIX_INPUT][TILE_WIDTH_MATRIX_INPUT];
//
//    // Coordinate calculations for input array for in-line unrolling
//    int c_in, h_in, w_in;
//    c_in = h / (K * K);
//    h_in = (h % (K * K)) / K + w / Width_out;
//    w_in = (h % (K * K)) % K + w % Width_out;
//
//    // Load shared memory tile
//    tile[ty][tx] = in_4d(b, c_in, h_in, w_in);
//    __syncthreads();
//
//    // Compute sum
//    float acc = 0.;
//    for (int p = 0; p < TILE_HEIGHT_MATRIX_INPUT; p++) {
//        acc += tile[p][tx] * mask_2d(h, p);
//    }
//    __syncthreads();
//
//    // Save sum to output
//    if (h < Map_out && w < Width_out_unrolled) {
//        out_3d(b, h, w) = acc;
//    }
//
//#undef in_4d
//#undef out_3d
//#undef mask_2d
//}
//
//__global__ void conv_forward_kernel_hidden_layer_matrix_unroll(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    /*
//    Function parameter definitions:
//    output - output
//    input - input
//    mask - convolution kernel
//    Batch - batch_size (number of images in x)
//    Map_out - number of output feature maps
//    Channel - number of input feature maps
//    Height - input height dimension
//    Width - input width dimension
//    K - kernel height and width (K x K)
//    */
//
//    // Compute output dimensions
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    const int Height_out_unrolled = (Channel) * (K * K);
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//
//    // Rename some useful variables
//    const int bx = blockIdx.x;
//    const int by = blockIdx.y;
//    const int bz = blockIdx.z;
//    const int tx = threadIdx.x;
//    const int ty = threadIdx.y;
//
//#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//#define out_3d(i2, i1, i0) output[(i2) * (Map_out * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
//#define mask_2d(i1, i0) cached_mask[(i1) * (Channel * K * K) + (i0)]
//
//    int w = bx * blockDim.x + tx; // Col of unrolled matrix
//    int h = by * blockDim.y + ty; // Row of unrolled matrix --> Col of mask
//    int b = bz;
//
//    __shared__ float tile[TILE_HEIGHT_MATRIX_HIDDEN][TILE_WIDTH_MATRIX_HIDDEN];
//
//    const int iters = Height_out_unrolled / TILE_HEIGHT_MATRIX_HIDDEN; // Number of iterations to compute one element
//    int height_offset, h_plus_offset, c_in, h_in, w_in;
//    float acc = 0.;
//    for (int i = 0; i < iters; i++) {
//        // Calculate height for iteration
//        height_offset = i * TILE_HEIGHT_MATRIX_HIDDEN;
//        h_plus_offset = h + height_offset;
//
//        // Coordinate calculations for input array for in-line unrolling
//        c_in = h_plus_offset / (K * K);
//        h_in = (h_plus_offset % (K * K)) / K + w / Width_out;
//        w_in = (h_plus_offset % (K * K)) % K + w % Width_out;
//
//        // Load shared memory tile
//        tile[ty][tx] = in_4d(b, c_in, h_in, w_in);
//        __syncthreads();
//
//        // Compute partial sum
//        for (int p = 0; p < TILE_HEIGHT_MATRIX_HIDDEN; p++) {
//            acc += tile[p][tx] * mask_2d(h, p + height_offset);
//        }
//        __syncthreads();
//    }
//
//    // Save sum to output
//    if (h < Map_out && w < Width_out_unrolled) {
//        out_3d(b, h, w) = acc;
//    }
//
//#undef in_4d
//#undef out_3d
//#undef mask_2d
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Allocate memory and copy over the relevant data structures to the GPU
//
//    // We pass double pointers for you to initialize the relevant device pointers,
//    //  which are passed to the other two functions.
//
//    // Useful snippet for error checking
//    // cudaError_t error = cudaGetLastError();
//    // if(error != cudaSuccess)
//    // {
//    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//    //     exit(-1);
//    // }
//
//
//    // Allocate GPU memory for input array
//    size_t sizeInput = Batch * Channel * Height * Width * sizeof(float);
//    cudaMalloc((void **) device_input_ptr, sizeInput);
//
//    // Calculate and allocate GPU memory needed for output array
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//    cudaMalloc((void **) device_output_ptr, sizeOutput);
//
//    // Allocate memory for mask filter (might need to change for constant memory)
//    size_t sizeMask = Map_out * Channel * K * K * sizeof(float);
//
//    // Transfer the input and mask to GPU
//    cudaMemcpy(*device_input_ptr, host_input, sizeInput, cudaMemcpyHostToDevice);
//    cudaMemcpyToSymbol(cached_mask, host_mask, sizeMask);
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Output dimensions
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//    // Unrolled dimensions and memory allocation
//    const int Height_out_unrolled = (Channel) * (K * K);
//    const int Width_out_unrolled = (Height_out) * (Width_out);
//
//    const int GridZ = Batch;
//
//    if (Height_out == 80 && Width_out == 80) {
//        // Use kernel for input layer
//
//        const int GridX = (Width_out_unrolled / TILE_WIDTH_MATRIX_INPUT);
//        const int GridY = (Height_out_unrolled / TILE_HEIGHT_MATRIX_INPUT);
//
//        dim3 DimGrid(GridX, GridY, GridZ);
//        dim3 DimBlock(TILE_WIDTH_MATRIX_INPUT, TILE_HEIGHT_MATRIX_INPUT, 1);
//
//        conv_forward_kernel_input_layer_matrix_unroll<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//        cudaDeviceSynchronize();
//
//    } else if (Height_out == 34 && Width_out == 34) {
//        // Use kernel for hidden layer
//
//        const int GridX = (Width_out_unrolled / TILE_WIDTH_MATRIX_HIDDEN);
//        const int GridY = (Height_out_unrolled / TILE_HEIGHT_MATRIX_HIDDEN);
//
//        dim3 DimGrid(GridX, GridY, GridZ);
//        dim3 DimBlock(TILE_WIDTH_MATRIX_HIDDEN, TILE_HEIGHT_MATRIX_HIDDEN, 1);
//
//        conv_forward_kernel_hidden_layer_matrix_unroll<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//        cudaDeviceSynchronize();
//
//    } else {
//        printf("I fell into an exception!\n");
//    }
//
//}
//
//
//__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
//{
//    // Copy the output back to host
//    const int Height_out = Height - K + 1;
//    const int Width_out = Width - K + 1;
//
//    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
//
//    cudaMemcpy(host_output, device_output, sizeOutput, cudaMemcpyDeviceToHost);
//
//    // Free device memory
//    cudaFree(device_input);
//    cudaFree(device_output);
//}
//
//
//__host__ void GPUInterface::get_device_properties()
//{
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    for(int dev = 0; dev < deviceCount; dev++)
//    {
//        cudaDeviceProp deviceProp;
//        cudaGetDeviceProperties(&deviceProp, dev);
//
//        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//    }
//}
// END: OPTIMIZATION 5 //

// START: OPTIMIZATION 6 //
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH_INPUT 16
#define TILE_WIDTH_HIDDEN 17
#define MASK_WIDTH 7
#define CHANNELS_PER_THREAD 2 // 4 does not fit
__constant__ float cached_mask[3136];

__global__ void conv_forward_kernel_input_layer(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) cached_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x; // Output map feature index
    const int W_size = Width_out / TILE_WIDTH_INPUT; // Number of horizontal tiles
    int first_output_row = (blockIdx.y / W_size) * TILE_WIDTH_INPUT;
    int first_output_col = (blockIdx.y % W_size) * TILE_WIDTH_INPUT;
    int h = first_output_row + threadIdx.y; // Row
    int w = first_output_col + threadIdx.x; // Column
    int b = blockIdx.z; // Batch element index


    // Load tile into shared memory
    const int input_tile_size = TILE_WIDTH_INPUT + K - 1; // = 22
    __shared__ float tile[1][22][22];
    int load_iters = ceil( (float)  (input_tile_size*input_tile_size*Channel) / (TILE_WIDTH_INPUT*TILE_WIDTH_INPUT));
    int one_dim_index_out, c, k, l;
    for (int i = 0; i < load_iters; i++){
        one_dim_index_out = threadIdx.y * TILE_WIDTH_INPUT + threadIdx.x + (i * TILE_WIDTH_INPUT * TILE_WIDTH_INPUT); // One dimensional index for output tile element
//        c = one_dim_index_out / (input_tile_size * input_tile_size); // Input channel --> Ignore for input layer
//        one_dim_index_out = one_dim_index_out - c * input_tile_size * input_tile_size; // Lower dimensionality of index
        k = one_dim_index_out / input_tile_size; // Input row
        l = one_dim_index_out % input_tile_size; // Input col
        if (k < input_tile_size && l < input_tile_size){
            tile[0][k][l] = in_4d(b, 0, first_output_row + k, first_output_col + l);
        }
    }
    __syncthreads();

    float acc = 0.0f; // Initialize sum

    for (int p = 0; p < K; p++) { // Loop over filter dimensions --> I am ignoring the channel dimension because the input layer only has one channel
        for (int q = 0; q < K; q++) {
            acc += tile[0][threadIdx.y + p][threadIdx.x + q] * mask_4d(m, 0, p, q);
        }
    }

    out_4d(b, m, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_hidden_layer(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) cached_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x; // Output map feature index
    const int W_size = Width_out / TILE_WIDTH_HIDDEN; // Number of horizontal tiles
    int first_output_row = (blockIdx.y / W_size) * TILE_WIDTH_HIDDEN;
    int first_output_col = (blockIdx.y % W_size) * TILE_WIDTH_HIDDEN;
    int h = first_output_row + threadIdx.y; // Row
    int w = first_output_col + threadIdx.x; // Column
    int b = blockIdx.z; // Batch element index

    // Load tile into shared memory
    const int input_tile_size = TILE_WIDTH_HIDDEN + K - 1;
    __shared__ float tile[4][23][23];
    int load_iters = ceil( (float)  (input_tile_size*input_tile_size*CHANNELS_PER_THREAD) / (TILE_WIDTH_HIDDEN*TILE_WIDTH_HIDDEN));
    int one_dim_index_out, c, k, l;
    for (int i = 0 ; i < load_iters; i++){
        one_dim_index_out = threadIdx.y * TILE_WIDTH_HIDDEN * (Channel / CHANNELS_PER_THREAD) + threadIdx.x * (Channel / CHANNELS_PER_THREAD) + threadIdx.z + (i * TILE_WIDTH_HIDDEN * TILE_WIDTH_HIDDEN * (Channel / CHANNELS_PER_THREAD)); // One dimensional index for output tile element
        c = one_dim_index_out / (input_tile_size * input_tile_size); // Input channel
        one_dim_index_out = one_dim_index_out - c * input_tile_size * input_tile_size; // Lower dimensionality of index
        k = one_dim_index_out / input_tile_size; // Input row
        l = one_dim_index_out % input_tile_size; // Input col
        if (c < Channel && k < input_tile_size && l < input_tile_size){
            tile[c][k][l] = in_4d(b, c, first_output_row + k, first_output_col + l);
        }
    }
    __syncthreads();


    // Use atomics for reduction over Channel dimension
    float acc = 0.; // Initialize sum for the channels computed by this thread only
    const int channel_offset = threadIdx.z * CHANNELS_PER_THREAD;
    for (int c = channel_offset; c < CHANNELS_PER_THREAD + channel_offset; c++) { // Sum across input channels
        for (int p = 0; p < K; p++) { // Loop over filter dimensions
            for (int q = 0; q < K; q++) {
                acc += tile[c][threadIdx.y + p][threadIdx.x + q] * mask_4d(m, c, p, q);
            }
        }
    }

    // Atomic reduction

    atomicAdd( &out_4d(b, m, h, w), acc);

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

    // Allocate GPU memory for input array
    size_t sizeInput = Batch * Channel * Height * Width * sizeof(float);
    cudaMalloc((void **) device_input_ptr, sizeInput);

    // Calculate and allocate GPU memory needed for output array
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMalloc((void **) device_output_ptr, sizeOutput);

    // Allocate memory for mask filter (might need to change for constant memory)
    size_t sizeMask = Map_out * Channel * K * K * sizeof(float);

    // Transfer the input and mask to GPU
    cudaMemcpy(*device_input_ptr, host_input, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cached_mask, host_mask, sizeMask);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int GridX = Map_out;
    const int GridZ = Batch;

    if (Height_out == 80 && Width_out == 80) {
        // Use kernel for input layer
        const int GridY = (Height_out/TILE_WIDTH_INPUT) * (Width_out/TILE_WIDTH_INPUT);

        dim3 DimGrid(GridX, GridY, GridZ);
        dim3 DimBlock(TILE_WIDTH_INPUT, TILE_WIDTH_INPUT, 1);

        conv_forward_kernel_input_layer<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
        cudaDeviceSynchronize();

    } else if (Height_out == 34 && Width_out == 34) {
        // Use kernel for hidden layer
        const int GridY = (Height_out/TILE_WIDTH_HIDDEN) * (Width_out/TILE_WIDTH_HIDDEN);

        dim3 DimGrid(GridX, GridY, GridZ);
        dim3 DimBlock(TILE_WIDTH_HIDDEN, TILE_WIDTH_HIDDEN, Channel / CHANNELS_PER_THREAD);

        conv_forward_kernel_hidden_layer<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
        cudaDeviceSynchronize();

    } else {
        printf("I fell into an exception!\n");
    }

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    size_t sizeOutput = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMemcpy(host_output, device_output, sizeOutput, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
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
// END: OPTIMIZATION 6 //