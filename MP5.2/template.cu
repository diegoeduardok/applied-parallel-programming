// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void create_auxiliary(float *input, float *output, int len_input, int len_output) {
  // Note to self: input is the output from the first scan and output is the auxiliary array.
  // This kernel is executed with one block only.

  // Rename some variables
  int tx = threadIdx.x;

  // Create auxiliary array with the last element from each (2*BLOCK_SIZE)-length section of scan's output.
  int offset = 2 * BLOCK_SIZE;
  int index = offset * (tx + 1) - 1;
  if (index < len_input) {

    output[tx] = input[index];

  } else if (index == (offset * len_output - 1)){
    // If the thread corresponds to the last element of the auxiliary array, find the last element of the input array
    // and load that one instead
    output[tx] = input[len_input - 1];
  }

  __syncthreads(); // Memory fence?

}

__global__ void add_auxiliary(float *output, float *aux, int len) {

  // Rename some relevant variables
  int tx = threadIdx.x;
  int bx = blockIdx.x + 1;

  int first_idx = tx + (bx * BLOCK_SIZE) * 2;
  int second_idx = first_idx + BLOCK_SIZE;

  float aux_element = aux[bx-1];

  if (first_idx < len) {
    output[first_idx] += aux_element;
    if (second_idx < len) {
      output[second_idx] += aux_element;
    }
  }
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  // Rename some relevant variables
  int tx = threadIdx.x;
  int bx = blockIdx.x;

  // Load input into shared memory
  __shared__ float partial[2*BLOCK_SIZE];

  int first_idx = tx + (bx * BLOCK_SIZE) * 2;
  int second_idx = first_idx + BLOCK_SIZE;

  if (first_idx < len) {
    partial[tx] = input[first_idx];
    if (second_idx < len) {
      partial[tx + BLOCK_SIZE] = input[second_idx];
    } else {
      partial[tx + BLOCK_SIZE] = 0.0f;
    }
  } else {
    partial[tx] = 0.0f;
    partial[tx + BLOCK_SIZE] = 0.0f;
  }
  __syncthreads();

  // Scan step
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
      partial[index] += partial[index-stride];
    }
    stride *= 2;
  }

  // Post scan step
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * BLOCK_SIZE){
      partial[index + stride] += partial[index];
    }
    stride /= 2;
  }

  __syncthreads();

  // Write output
  if (first_idx < len) {
    output[first_idx] = partial[tx];
    if (second_idx < len) {
      output[second_idx] = partial[tx + BLOCK_SIZE];
    }
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int GridX = ceil((float) numElements / (2*BLOCK_SIZE));
  dim3 GridDim(GridX, 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1, 1);

  // Assign space for auxiliary array and scanned auxiliary array in global memory
  float *deviceAux;
  float *deviceAuxScan;
  wbCheck(cudaMalloc((void **)&deviceAux, GridX * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxScan, GridX * sizeof(float)));


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  scan<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, numElements); // Initial scan

  // Change dimensions to a single block for second scan
  create_auxiliary<<<1, BlockDim>>>(deviceOutput, deviceAux, numElements, GridX);
  scan<<<1, BlockDim>>>(deviceAux, deviceAuxScan, GridX); // Second scan

  dim3 GridDim2(GridX-1, 1, 1);
  add_auxiliary<<<GridDim2, BlockDim>>>(deviceOutput, deviceAuxScan, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAux); // Added by student
  cudaFree(deviceAuxScan); // Added by student
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
