#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/sgemm_traits.h"

#include "gemm_helper.h"
#include "timing_helper.h"

cudaError_t CutlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc,
    cudaStream_t stream,
    int n_kernel
    )
{

  typedef cutlass::gemm::SgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::Shape<4*SCALE_K, RATIO*SCALE_N, RATIO*SCALE_M>,
      cutlass::gemm::LinearScaling<float>,
      cutlass::Shape<4*SCALE_K, 1*SCALE_N, 1*SCALE_M>
      >
      GemmTraits;

  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

  typename Gemm::Params params;

  int result = params.initialize(
      M,
      N,
      K,
      alpha,
      A,
      lda,
      B,
      ldb,
      beta,
      C,
      ldc,
      C,
      ldc);

  if (result)
  {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }

  for (int i=0; i<n_kernel; i++)
    Gemm::launch(params, stream);

  return cudaGetLastError();
}

cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta, int n_stream, int n_kernel_per_stream)
{
  Timer timer;

  int lda = M;
  int ldb = K;
  int ldc = M;

  float *A;
  float *B;
  float *C_cutlass;

  AllocateMatrix(&A, lda, M, K, /* seed */ 0);
  AllocateMatrix(&B, ldb, K, N, /* seed */ 17);
  AllocateMatrix(&C_cutlass, ldc, M, N, /* seed */ 101);

  cudaStream_t streams[n_stream];
  for (int i=0; i<n_stream; i++)
    cudaStreamCreate(&streams[i]);
  
    
  cudaDeviceSynchronize();
  timer.start();
  for (int i=0; i<n_stream; i++)
    CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, streams[i], n_kernel_per_stream);
  
  cudaDeviceSynchronize();
  long long duration = timer.finish_and_get_us();
  std::cout << "us: " << duration << std::endl;

  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaGetLastError();
}

int main(int argc, const char *arg[])
{

  int problem[3] = {128, 128, 128};
  float scalars[2] = {1, 1};
  int config[2] = {1, 1};

  // Problems parsing M, N, K
  for (int i = 1; i < argc && i < 4; ++i)
  {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Config parsing n_stream, n_kernel_per_stream
  for (int i = 4; i < argc && i < 6; ++i)
  {
    std::stringstream ss(arg[i]);
    ss >> config[i - 4];
  }

  cudaError_t result = TestCutlassGemm(
      problem[0],
      problem[1],
      problem[2],
      scalars[0],
      scalars[1],
      config[0],
      config[1]
      );

  return result == cudaSuccess ? 0 : -1;
}
