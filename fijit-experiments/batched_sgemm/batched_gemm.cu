#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/sgemm_traits.h"

#include "../gemm_helper.h"
#include "../timing_helper.h"

cudaError_t cutlass_strided_batched_sgemm(float const *A,
  int lda,
  long long int batch_stride_A,
  float const *B,
  int ldb,
  long long int batch_stride_B,
  float *C,
  int ldc,
  long long int batch_stride_C,
  float alpha,
  float beta,
  int m, 
  int n,
  int k,
  int batch_count,
  cudaStream_t stream,
  int n_kernel_per_stream
  ) 
  {
  // create a cutlass traits
  typedef cutlass::gemm::SgemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, 
    cutlass::Shape<8, 128, 128> 
    >
    SgemmTraits;
  
  // create a CUTLASS GEMM object.
  typedef cutlass::gemm::Gemm<SgemmTraits> Gemm;
  
  // Construct and initialize CUTLASS GEMM parameters object.
  typename Gemm::Params params;
  
  int result = params.initialize(
    m,                  // M dimension for each batch
    n,                  // N dimension for each batch
    k,                  // K dimension for each batch
    alpha,              // scalar alpha
    A,
    lda,
    batch_stride_A,     // distance in memory between the first element of neighboring batch
    B,
    ldb,
    batch_stride_B,     // distance in memory between the first element of neighboring batch
    beta,               // scalar beta
    C,                  // source matrix C
    ldc,
    batch_stride_C,     // distance in memory between the first element of neighboring batch
    C,                  // destination matrix C (may be different memory than source C matrix)
    ldc,
    batch_stride_C,    // distance in memory between the first element of neighboring batch
    batch_count
  );
  
  if (result != 0) {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }
  
  for (int i=0; i<n_kernel_per_stream; i++)
    Gemm::launch(params, stream);

  return cudaGetLastError();
}

int main(int argc, const char *arg[]) {
  /*int const partition_cnt = 5;*/
  int const im_batch_size = 1;
  int const im_C = 128;
  int const im_H = 28;
  int const im_W = 28;
  int const k_H = 3;
  int const k_W = 3;
  int const out_channel = 128;

  int const inp_tile = (im_H - 2) * (im_W - 2) / 4;
  static_assert(k_H == 3 && k_W == 3, "Only works for 3x3 kernel");

  int const m = inp_tile;
  int const n = out_channel;
  int const k = im_C;
  int const batch_count = 16 * im_batch_size;

  // Config parsing n_stream, n_kernel_per_stream
  int config[2] = {1,1};
  for (int i = 1; i < argc && i < 3; ++i)
  {
    std::stringstream ss(arg[i]);
    ss >> config[i - 1];
  }
  int n_stream = config[0];
  int n_kernel_per_stream = config[1];

  // A, B are non-transpose, column major
  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  int const count_A = batch_count * lda * k;
  int const count_B = ldb * n;
  int const count_C = batch_count * ldc * n;

  // the memory is batched along K dimension
  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

  // alpha and beta
  float alpha = 1.0f;
  float beta = 2.0f;

  cudaError_t result = cudaSuccess;

  // allocate the host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  std::vector<float> result_C(count_C);

  // allocate the device memory
  float *A;
  float *B;
  float *C;

  cudaMalloc(&A, count_A * sizeof(float));
  cudaMalloc(&B, count_B * sizeof(float));
  cudaMalloc(&C, count_C * sizeof(float));

  // fill A
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < k; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_A[row_idx + col_idx * lda + b_idx * lda * k] = static_cast<float>(row_idx + col_idx * lda + b_idx * lda * k);
      }
    }
  }
  // fill B
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < k; row_idx++) {
        host_B[row_idx + col_idx * ldb + b_idx * k] = static_cast<float>(n + k * ldb + batch_count * k) - static_cast<float>(row_idx + col_idx * ldb + b_idx * k);
      }
    }
  }
  // fill C
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_C[row_idx + col_idx * ldc + b_idx * ldc * n] = 1.f;
      }
    }
  }

  // ref memory
  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  // copy host memory to device
  cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice);

  // run cutlass
  cudaStream_t streams[n_stream]
  Timer timer
  for (int i=0; i<n_stream; i++)
    cudaStreamCreate(&streams[i]);

  cudaDeviceSynchronize();
  timer.start();

  for (int i=0;i<n_stream;i++)
    cutlass_strided_batched_sgemm(A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C,
      alpha, beta, m, n, k, batch_count, streams[i], n_kernel_per_stream);

  cudaDeviceSynchronize();
  long long duration = timer.finish_and_get_us();
  std::cout << "us: " << duration << std::endl;

  
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  // Exit.
  return cudaSuccess == cudaGetLastError() ? 0 : -1;
}
