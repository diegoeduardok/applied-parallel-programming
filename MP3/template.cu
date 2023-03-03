
#include <wb.h>

#define TILE_WIDTH 32

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty; // row in C matrix
  int col = bx * TILE_WIDTH + tx; // col in C matrix
  float Pval = 0.;

  int width = numAColumns; // = numBRows
  for (int q = 0; q < ceil((float)width/TILE_WIDTH); ++q){ // q iterates over tiles across A cols and B rows
      // Load sub-tile for A
      if (row < numARows && (q*TILE_WIDTH+tx) < width){ // row has to be less than A rows and col has to be less than A cols
          subTileA[ty][tx] = A[row * width + q * TILE_WIDTH + tx]; // iterate over columns of A for qth sub-tile
      } else{
          subTileA[ty][tx] = 0;
      }
      // Load sub-tile for B
      if (col < numBColumns && (q*TILE_WIDTH+ty) < width){ // col has to be less than B cols and row has to be less than B rows
          subTileB[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + col]; // iterate over rows of B for qth sub-tile
      } else{
          subTileB[ty][tx] = 0;
      }
      __syncthreads();
      // Perform partial summation
      for (int i = 0; i < TILE_WIDTH; ++i){
          Pval += subTileA[ty][i] * subTileB[i][tx]; // add numbers in qth sub-tile
      }
      __syncthreads();
  }
  // Assign value if within bounds
    if (row < numCRows && col < numCColumns){
        C[row * numCColumns + col] = Pval;
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  size_t sizeC = sizeof(float)*numCRows*numCColumns;
  hostC = (float *)malloc(sizeC);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  size_t sizeA = sizeof(float) * numARows * numAColumns;
  size_t sizeB = sizeof(float) * numBRows * numBColumns;
  cudaMalloc((void **)&deviceA, sizeA);
  cudaMalloc((void **)&deviceB, sizeB);
  cudaMalloc((void **)&deviceC, sizeC);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int GridXDim = ceil((float)numBColumns / TILE_WIDTH);
  int GridYDim = ceil((float)numARows / TILE_WIDTH);
  dim3 DimGrid(GridXDim, GridYDim, 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
