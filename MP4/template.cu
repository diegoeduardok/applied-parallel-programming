#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define MASK_WIDTH 3
#define MASK_RADIUS ((MASK_WIDTH - 1) / 2)
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  // Initialize shared tile
  __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

//  printf("kernel: %1.5f\n", deviceKernel[0]);

  // Set output coordinates
  int z_out = blockIdx.z * TILE_WIDTH + tz;   // We can't use blockDim.z here because now TILE_WIDTH !=
                                              // blockDim.{x, y, z}
  int y_out = blockIdx.y * TILE_WIDTH + ty;
  int x_out = blockIdx.x * TILE_WIDTH + tx;

  // Set input coordinates
  int z_in = z_out - MASK_RADIUS;
  int y_in = y_out - MASK_RADIUS;
  int x_in = x_out - MASK_RADIUS;

  // Initialize variable to store output
  float result = 0.0;
//  float kernel;

//  printf("Thread: (%d, %d, %d) | Result: %1.2f\n", tz, ty, tx, result);

  // Load tile into shared memory
  bool z_in_bounds = (z_in >= 0) && (z_in < z_size);
  bool y_in_bounds = (y_in >= 0) && (y_in < y_size);
  bool x_in_bounds = (x_in >= 0) && (x_in < x_size);
//  printf("Thread: (%d, %d, %d) | z_in_bounds val: %d\n", tz, ty, tx, z_in_bounds);

  if(z_in_bounds && y_in_bounds && x_in_bounds){
    tile[tz][ty][tx] = input[z_in * (y_size * x_size) + y_in * x_size + x_in];
  } else {
    tile[tz][ty][tx] = 0.0;
  }

  __syncthreads();

  // Compute output
  bool non_halo_thread = (tz < TILE_WIDTH) && (ty < TILE_WIDTH) && (tx < TILE_WIDTH);
  if(non_halo_thread){
    for(int i = 0; i < MASK_WIDTH; i++){
      for(int j = 0; j < MASK_WIDTH; j++){
        for(int k = 0; k < MASK_WIDTH; k++){
          result += deviceKernel[i][j][k] * tile[tz + i][ty + j][tx + k];
        }
      }
    }
    // Assign output
    bool non_ghost_thread = (z_out < z_size) && (y_out < y_size) && (x_out < x_size);
    if(non_ghost_thread){
      output[z_out * (y_size * x_size) + y_out * x_size + x_out] = result;
    }
  }

};

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
          (float *) wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *) malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions

  size_t sizeInput = (inputLength - 3) * sizeof(float);
  cudaMalloc((void **) &deviceInput, sizeInput);

  size_t sizeOutput = sizeInput;
  cudaMalloc((void **) &deviceOutput, sizeOutput);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu

  cudaMemcpy(deviceInput, &hostInput[3], sizeInput, cudaMemcpyHostToDevice);

  size_t sizeKernel = kernelLength * sizeof(float);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, sizeKernel);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  int GridXdim = ceil((float)x_size / TILE_WIDTH);
  int GridYdim = ceil((float)y_size / TILE_WIDTH);
  int GridZdim = ceil((float)z_size / TILE_WIDTH);

  dim3 DimGrid(GridXdim, GridYdim, GridZdim);
  dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

  //@@ Launch the GPU kernel here

  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)

  cudaMemcpy(&hostOutput[3], deviceOutput, sizeOutput, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
