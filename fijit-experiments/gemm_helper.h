#include <cuda_runtime_api.h>

__global__ void InitializeMatrix_kernel(
    float *matrix,
    int ldm,
    int rows,
    int columns,
    int seed = 0)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns)
  {
    int offset = i + j * ldm;

    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

cudaError_t InitializeMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0)
{

  dim3 block(16, 16);
  dim3 grid(
      (rows + block.x - 1) / block.x,
      (columns + block.y - 1) / block.y);

  InitializeMatrix_kernel<<<grid, block>>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

cudaError_t AllocateMatrix(float **matrix, int ldm, int rows, int columns, int seed = 0)
{
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * ldm * columns;

  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess)
  {
    std::cerr << "Failed to allocate matrix: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess)
  {
    std::cerr << "Failed to clear matrix device memory: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

  if (result != cudaSuccess)
  {
    std::cerr << "Failed to initialize matrix: "
              << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}