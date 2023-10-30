#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <iostream>


#include "stdio.h"

// for debugging
// #define CUDA_ERROR_CHECK
// #define CUDA_TIMING

#define cudaSafeCall( err ) __cnnCudaSafeCall( err, __FILE__, __LINE__ )

inline void __cnnCudaSafeCall( cudaError_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }
#endif
  return;
}


#ifdef CUDA_TIMING
class CudaTimer
{
public:
  CudaTimer() 
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CudaTimer() 
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() 
  {
    cudaEventRecord(start_, 0);
  }

  float elapsed() 
  {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float t = 0;
    cudaEventElapsedTime(&t, start_, stop_);
    return t;
  }

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
#endif

// CUDA kernels
template <typename T>
__global__ void cuda_logsumexp_pot_act_forward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> weight,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> mu,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> sigma,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> max_exponent,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> f,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> f_prime)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int ic = blockIdx.y * blockDim.y + threadIdx.y;

  const int C = x.size(0);
  const int N = x.size(1);

  if(ic >= C || ix >= N)
    return;

  const int Nw = weight.size(1);
  const T x_v = x[ic][ix];
  T s2 = sigma[ic] * sigma[ic];
  T loc_max_exponent = log(0.);
  for (auto i = 0; i < Nw; ++i)
  {
    const auto d = x_v - mu[i];
    const auto exponent = -(d * d) / (2 * s2);
    loc_max_exponent = max(loc_max_exponent, exponent);
  }

  T wexp = 0.;
  T wexpd = 0.;
  for (auto i = 0; i < Nw; ++i)
  {
    const auto d = x_v - mu[i];
    const auto exponent = -(d * d) / (2 * s2);
    static constexpr T sqrt_2pi = 2.5066282746310002;
    const auto w_exp = weight[ic][i] * exp(exponent - loc_max_exponent) / (sqrt_2pi * sigma[ic]);
    wexp += w_exp;
    wexpd += w_exp * d;
  }
  f[ic][ix] = -(log(wexp) + loc_max_exponent);
  f_prime[ic][ix] = wexpd / (wexp * s2);
  max_exponent[ic][ix] = loc_max_exponent;
}

template <typename T>
__global__ void cuda_logsumexp_pot_act_backward_kernel(
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> weight,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_out,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> grad_out_prime,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> mu,
  const torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> sigma,
  const torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> max_exponent,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> x_grad_out,
  torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits> w_grad_out,
  torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits> sigma_grad_out)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int ic = blockIdx.y * blockDim.y + threadIdx.y;

  const int C = x.size(0);
  const int N = x.size(1);

  if(ic >= C || ix >= N)
    return;

  const auto Nw = weight.size(1);

  const auto x_v = x[ic][ix];
  const auto grad_out_v = grad_out[ic][ix];
  const auto grad_out_prime_v = grad_out_prime[ic][ix];
  const auto max_exponent_v = max_exponent[ic][ix];

  T s2 = sigma[ic] * sigma[ic];
  T s3 = s2 * sigma[ic];

  T wexp = 0.;
  T wexpd = 0.;
  T wexpd2 = 0.;
  T wexpd3 = 0.;
  for (auto i = 0; i < Nw; ++i)
  {
    const auto d = x_v - mu[i];
    const auto d2 = d * d;
    const auto exponent = -d2 / (2 * s2);
    const auto w_exp = weight[ic][i] * exp(exponent - max_exponent_v);
    wexp += w_exp;
    wexpd += w_exp * d;
    wexpd2 += w_exp * d2;
    wexpd3 += w_exp * d * d2;
  }
  const T dxf = wexpd / (wexp * s2);
  const T dxdxf = 1 / s2 + dxf * dxf - (wexpd2 / wexp) / (s2 * s2);
  x_grad_out[ic][ix] = dxf * grad_out_v + dxdxf * grad_out_prime_v;

  const auto dsf = (s2 * wexp - wexpd2) / wexp / s3;
  const auto dsdxf = (wexpd3 / sigma[ic] * wexp - wexpd * (2 * sigma[ic] * wexp + wexpd2 / sigma[ic])) / (s2 * s2 * wexp * wexp);
  const T grad_sigma = dsf * grad_out_v + dsdxf * grad_out_prime_v;

  atomicAdd(&(sigma_grad_out[ic]), grad_sigma);

  for (auto i = 0; i < Nw; ++i)
  {
    const auto d = x_v - mu[i];
    const auto ee = exp(-(d * d) / (2 * s2) - max_exponent_v);
    const auto dwf = -ee / wexp;
    const auto dwdxf = dwf * (dxf - d / s2);
    T grad_w = dwf * grad_out_v + dwdxf * grad_out_prime_v;
    atomicAdd(&(w_grad_out[ic][i]), grad_w);
  }
}

std::vector<torch::Tensor> cuda_logsumexp_pot_act_forward(
  const torch::Tensor x,
  const torch::Tensor weight,
  const torch::Tensor mu,
  const torch::Tensor sigma)
{
  TORCH_CHECK(x.dim() == 4, "Expected 4d tensor as 1st input");
  TORCH_CHECK(weight.dim() == 2, "Expected 2d tensor as 2nd input");
  TORCH_CHECK(x.size(1) == weight.size(0), "Input channels must match weight channels");
  
  int B = x.size(0);
  int C = x.size(1);
  int H = x.size(2);
  int W = x.size(3);

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  // transform to channels first
  auto xc = torch::permute(x, {1, 0, 2, 3}).contiguous().reshape({C, B*H*W});

  auto out = torch::empty_like(xc);
  auto out_prime = torch::empty_like(xc);
  auto max_exponent = torch::empty({C, B*H*W}, xc.options());

  const dim3 blockSize(1024, 1, 1);
  const dim3 numBlocks((xc.size(1)+blockSize.x-1) / blockSize.x,
                       (xc.size(0)+blockSize.y-1) / blockSize.y);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "logsumexp_pot_act_forward", ([&]{
    cuda_logsumexp_pot_act_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      xc.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      weight.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      mu.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      sigma.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      max_exponent.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      out_prime.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
      );
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << "ms" << std::endl;
#endif

  return {
    out.reshape({C, B, H, W}).permute({1, 0, 2, 3}).contiguous(),
    out_prime.reshape({C, B, H, W}).permute({1, 0, 2, 3}).contiguous(),
    max_exponent
  };
}

std::vector<torch::Tensor> cuda_logsumexp_pot_act_backward(
  const torch::Tensor x,
  const torch::Tensor weight,
  const torch::Tensor grad_out,
  const torch::Tensor grad_out_prime,
  const torch::Tensor mu,
  const torch::Tensor sigma,
  const torch::Tensor max_exponent)
{
  TORCH_CHECK(x.dim() == 4, "Expected 4d tensor as 1st input");
  TORCH_CHECK(weight.dim() == 2, "Expected 2d tensor as 2nd input");
  TORCH_CHECK(x.size(1) == weight.size(0), "Input channels must match weight channels");

  int B = x.size(0);
  int C = x.size(1);
  int H = x.size(2);
  int W = x.size(3);

#ifdef CUDA_TIMING
  CudaTimer cut;
  cut.start();
#endif

  // transform to channels first
  auto xc = torch::permute(x, {1, 0, 2, 3}).contiguous().reshape({C, B*H*W});
  auto grad_outc = torch::permute(grad_out, {1, 0, 2, 3}).contiguous().reshape({C, B*H*W});
  auto grad_out_primec = torch::permute(
      grad_out_prime, {1, 0, 2, 3}
  ).contiguous().reshape({C, B*H*W});

  auto x_grad_out = torch::zeros_like(xc);
  auto w_grad_out = torch::zeros_like(weight);
  auto sigma_grad_out = torch::zeros_like(sigma);

  const dim3 blockSize(64, 16, 1);
  const dim3 numBlocks((xc.size(1)+blockSize.x-1) / blockSize.x,
                       (xc.size(0)+blockSize.y-1) / blockSize.y);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "logsumexp_pot_act_backward", ([&]{
    cuda_logsumexp_pot_act_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
      xc.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      weight.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      grad_outc.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      grad_out_primec.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      mu.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      sigma.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      max_exponent.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      x_grad_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      w_grad_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      sigma_grad_out.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
      );
  }));
  cudaSafeCall(cudaGetLastError());

#ifdef CUDA_TIMING
  cudaDeviceSynchronize();
  std::cout << "forward time " << cut.elapsed() << "ms" << std::endl;
#endif

  return {
    x_grad_out.reshape({C, B, H, W}).permute({1, 0, 2, 3}).contiguous(),
    w_grad_out,
    sigma_grad_out
  };
}
