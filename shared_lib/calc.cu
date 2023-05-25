#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <gmp.h>

#include "cgbn/cgbn.h"
#include "../samples/utility/cpu_support.h"
#include "../samples/utility/cpu_simple_bn_math.h"
#include "../samples/utility/gpu_support.h"


/*
   void cgbn_modular_power(cgbn_env_t env, cgbn_t &r, const cgbn_t &x, const cgbn_t &e, const cgbn_t &m)
   Computes r = x^e modulo the modulus, m. Requires that x < m.
 */


// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define BITS 4096
uint32_t Instances=10;

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> power;
  cgbn_mem_t<BITS> modulus;
  cgbn_mem_t<BITS> result;
} instance_t;

void myPrint(uint32_t *x, uint32_t count) {
  int32_t index;

  for(index=count-1;index>=0;index--)
    printf("%u ", x[index]);
  printf("\n");
}

// gnerate random input
void generate_random_instances(instance_t *instances, uint32_t count) {
  int         index;

  for(index=0;index<count;index++) {
    random_words(instances[index].x._limbs, BITS/32);
    random_words(instances[index].power._limbs, BITS/32);
    random_words(instances[index].modulus._limbs, BITS/32);
    zero_words(instances[index].result._limbs, BITS/32);

    // ensure modulus is odd
    instances[index].modulus._limbs[0] |= 1;

    // ensure modulus is greater than 
    if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, BITS/32)>0) {
      swap_words(instances[index].x._limbs, instances[index].modulus._limbs, BITS/32);

      // modulus might now be even, ensure it's odd
      instances[index].modulus._limbs[0] |= 1;
    }
    else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, BITS/32)==0) {
      // since modulus is odd and modulus = x, we can just subtract 1 from x
      instances[index].x._limbs[0] -= 1;
    }
  }
}


// support routine to verify the GPU results using the CPU
void verify_results(instance_t *instances, uint32_t count) {
  mpz_t x, p, m, computed, correct;

  mpz_init(x);
  mpz_init(p);
  mpz_init(m);
  mpz_init(computed);
  mpz_init(correct);

  uint32_t matched = 0;
  for(uint32_t index=0;index<count;index++) {
    printf("[%u]\n", index);
    to_mpz(x, instances[index].x._limbs, BITS/32);
    to_mpz(p, instances[index].power._limbs, BITS/32);
    to_mpz(m, instances[index].modulus._limbs, BITS/32);
    to_mpz(computed, instances[index].result._limbs, BITS/32);

    mpz_powm(correct, x, p, m);

    //printf("result:\n");
    //myPrint(instances[index].result._limbs, BITS/32);

    //printf("answer:\n");
    uint32_t c[BITS/32] = {0};
    from_mpz(correct, c, BITS/32);
    //myPrint(c, BITS/32);

    if(mpz_cmp(correct, computed)!=0) {
      printf("gpu inverse kernel failed on instance %d\n", index);
    }
    else matched++;
  }

  mpz_clear(x);
  mpz_clear(p);
  mpz_clear(m);
  mpz_clear(computed);
  mpz_clear(correct);

  printf("===> %u / %u results match.\n", matched, count);
}

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// the actual kernel
__global__ void kernel_modular_power(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 4096-bit math
  env_t::cgbn_t  r, x, p, m;                                          // define r, x, e, m as 4096-bit bignums

  cgbn_load(bn_env, x, &(instances[instance].x));            // load my instance's x value
  cgbn_load(bn_env, p, &(instances[instance].power));        // load my instance's p value
  cgbn_load(bn_env, m, &(instances[instance].modulus));      // load my instance's m value
  cgbn_modular_power(bn_env, r, x, p, m);                    // r = x^p % m
  cgbn_store(bn_env, &(instances[instance].result), r);      // store r into sum
}

// mpz_powm gmp on cpu
void powm_gmp(instance_t *instances, uint32_t count) {
  mpz_t r, x, p, m;
  uint32_t c_buf[BITS/32] = {0};

  mpz_init(r);
  mpz_init(x);
  mpz_init(p);
  mpz_init(m);

  for(int index=0;index<count;index++) {
    to_mpz(x, instances[index].x._limbs, BITS/32);
    to_mpz(p, instances[index].power._limbs, BITS/32);
    to_mpz(m, instances[index].modulus._limbs, BITS/32);

    mpz_powm(r, x, p, m);
    from_mpz(r, c_buf, BITS/32);
  }

  mpz_clear(r);
  mpz_clear(x);
  mpz_clear(p);
  mpz_clear(m);
}

// mpz_powm gmp openmp on cpu
void powm_gmp_omp(instance_t *instances, uint32_t count) {
  mpz_t r, x, p, m;
  uint32_t c_buf[BITS/32] = {0};

  mpz_init(r);
  mpz_init(x);
  mpz_init(p);
  mpz_init(m);

  #pragma omp parallel for
  for(int index=0;index<count;index++) {
    //printf("[%d]\n", index);
    to_mpz(x, instances[index].x._limbs, BITS/32);
    to_mpz(p, instances[index].power._limbs, BITS/32);
    to_mpz(m, instances[index].modulus._limbs, BITS/32);

    mpz_powm(r, x, p, m);

    //printf("result:\n");
    from_mpz(r, c_buf, BITS/32);
    //myPrint(c_buf, BITS/32);
  }

  mpz_clear(r);
  mpz_clear(x);
  mpz_clear(p);
  mpz_clear(m);
}


extern "C" void cgbn_power(int size) {
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;

  Instances = size;
  instances = (instance_t *)malloc(sizeof(instance_t)*Instances);
  generate_random_instances(instances, Instances);
  cgbn_error_report_alloc(&report);

  cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*Instances);
  cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*Instances, cudaMemcpyHostToDevice);

  kernel_modular_power<<<(Instances+3)/4, 128>>>(report, gpuInstances, Instances);
  cudaDeviceSynchronize();

  cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*Instances, cudaMemcpyDeviceToHost);

  // clean up
  free(instances);
  cudaFree(gpuInstances);
  cgbn_error_report_free(report);
}


extern "C" void cgbn_power_uma(int size) {
  instance_t          *instances;
  cgbn_error_report_t *report;

  Instances = size;
  cgbn_error_report_alloc(&report);
  cudaMallocManaged((void **)&instances, sizeof(instance_t)*Instances);
  generate_random_instances(instances, Instances);

  kernel_modular_power<<<(Instances+3)/4, 128>>>(report, instances, Instances);
  cudaDeviceSynchronize();

  // clean up
  cudaFree(instances);
  cgbn_error_report_free(report);
}


extern "C" void cgbn_power_zc(int size) {
  instance_t          *h_instances, *d_instances;
  cgbn_error_report_t *report;
  
  Instances = size;
  cgbn_error_report_alloc(&report);
  cudaHostAlloc((void **)&h_instances, sizeof(instance_t)*Instances, cudaHostAllocMapped);
  cudaHostGetDevicePointer((void **)&d_instances, (void *)h_instances, 0);
  generate_random_instances(h_instances, Instances);

  kernel_modular_power<<<(Instances+3)/4, 128>>>(report, d_instances, Instances);
  cudaDeviceSynchronize();


  // clean up
  cudaFree(d_instances);
  cudaFreeHost(h_instances);
  cgbn_error_report_free(report);
}



extern "C" void omp_gmp_power(int size) {
  instance_t          *instances;

  Instances = size;
  instances = (instance_t *)malloc(sizeof(instance_t)*Instances);
  generate_random_instances(instances, Instances);

  powm_gmp_omp(instances, Instances);

  // clean up
  free(instances);
}


extern "C" void gmp_power(int size) {
  instance_t          *instances;
  Instances = size;
  instances = (instance_t *)malloc(sizeof(instance_t)*Instances);
  generate_random_instances(instances, Instances);

  powm_gmp(instances, Instances);

  // clean up
  free(instances);
}

int main() {
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  uint64_t            startTime, endTime;

  printf("Genereating instances: %u ...\n", Instances);
  instances = (instance_t *)malloc(sizeof(instance_t)*Instances);
  generate_random_instances(instances, Instances);

  printf("Copying instances to the GPU ...\n");
  startTime = std::chrono::duration_cast< std::chrono::microseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*Instances));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*Instances, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  printf("Running GPU kernel ...\n");
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  kernel_modular_power<<<(Instances+3)/4, 128>>>(report, gpuInstances, Instances);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  //CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*Instances, cudaMemcpyDeviceToHost));
  endTime = std::chrono::duration_cast< std::chrono::microseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
  printf("kernel_modular_power %llu us \n", endTime - startTime);

  //printf("Verifying the results ...\n");
  //verify_results(instances, Instances);

  printf("\n");
  printf("Running GMP powm with OpenMP ...\n");
  startTime = std::chrono::duration_cast< std::chrono::microseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
  powm_gmp_omp(instances, Instances);
  endTime = std::chrono::duration_cast< std::chrono::microseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
  printf("powm_gmp_omp %llu us \n", endTime - startTime);

  printf("\n");
  printf("Running GMP powm ...\n");
  startTime = std::chrono::duration_cast< std::chrono::microseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
  powm_gmp(instances, Instances);
  endTime = std::chrono::duration_cast< std::chrono::microseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
  printf("powm_gmp %llu us \n", endTime - startTime);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));

  printf("Performance test done ...\n");
  return 0;
}
