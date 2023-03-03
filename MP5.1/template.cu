// MP 5.1 Reduction
// Given a list of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

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
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  __shared__ float partial_sum[2*BLOCK_SIZE];

  unsigned int tx = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  unsigned int input_idx = start + tx;
  unsigned int second_input_idx; // No need to compute if input_idx >= len

  if (input_idx < len){
    partial_sum[tx] = input[input_idx];
    second_input_idx = input_idx + blockDim.x;
    if (second_input_idx < len){
      partial_sum[tx + blockDim.x] = input[second_input_idx];
    } else {
      partial_sum[tx + blockDim.x] = 0.0f;
    }
  } else {
    partial_sum[tx] = 0.0f;
    partial_sum[tx + blockDim.x] = 0.0f;
  }

  //@@ Traverse the reduction tree
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
    __syncthreads();
    if (tx < stride){
      partial_sum[tx] += partial_sum[tx + stride];
    }
  }

  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  output[blockIdx.x] = partial_sum[0];
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = (numInputElements - 1) / (BLOCK_SIZE << 1) + 1;
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  size_t size_input_bytes = numInputElements * sizeof(float);
  cudaMalloc((void **) &deviceInput, size_input_bytes);
  size_t size_output_bytes = numOutputElements * sizeof(float);
  cudaMalloc((void **) &deviceOutput, size_output_bytes);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, size_input_bytes, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  int GridDimX = ceil((float) numInputElements / (2*BLOCK_SIZE));
  dim3 GridDim(GridDimX, 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1 ,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, numInputElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, size_output_bytes, cudaMemcpyDeviceToHost);


  wbTime_stop(Copy, "Copying output memory to the CPU");

  /***********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input.
   * For simplicity, we do not require that for this lab!
   ***********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
