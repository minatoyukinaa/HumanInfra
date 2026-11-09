#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void elementwise_add_f16(half *a, half *b, half *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main(){
    const int    N     = 1 << 24;                 // 16M
    const size_t bytes = (size_t)N * sizeof(half);

    half *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    for (int i = 0; i < N; ++i) {
        a[i] = __float2half((float)(i % 100) * 0.01f);
        b[i] = __float2half((float)(i %  50) * 0.02f);
    }

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    elementwise_add_f16<<<grid, block>>>(a, b, c, N);   // warm-up：触发页迁移
    cudaDeviceSynchronize();

    elementwise_add_f16<<<grid, block>>>(a, b, c, N);   // 被 profile 的目标
    cudaDeviceSynchronize();

    // 打印首尾元素，确认正确性 & 防止 DCE
    printf("c[0]=%f  c[N-1]=%f\n",
           __half2float(c[0]), __half2float(c[N-1]));

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}