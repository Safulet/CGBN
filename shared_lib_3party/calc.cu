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
#define BITS 2048
#define window_bits 5

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

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
		//printf("[%u]\n", index);
		to_mpz(x, instances[index].x._limbs, BITS/32);
		to_mpz(p, instances[index].power._limbs, BITS/32);
		to_mpz(m, instances[index].modulus._limbs, BITS/32);
		to_mpz(computed, instances[index].result._limbs, BITS/32);

		/*
		   printf("x:\n");
		   myPrint(instances[index].x._limbs, BITS/32);
		   printf("p:\n");
		   myPrint(instances[index].power._limbs, BITS/32);
		   printf("m:\n");
		   myPrint(instances[index].modulus._limbs, BITS/32);
		   printf("result:\n");
		   myPrint(instances[index].result._limbs, BITS/32);
		 */
		mpz_powm(correct, x, p, m);

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


__device__ __forceinline__ void fixed_window_powm_odd(env_t _env, env_t::cgbn_t &result, env_t::cgbn_t &x, env_t::cgbn_t &power, env_t::cgbn_t &modulus) {
	env_t::cgbn_t       t;
	env_t::cgbn_local_t window[1<<window_bits];
	int32_t    index, position, offset;
	uint32_t   np0;

	// conmpute x^power mod modulus, using the fixed window algorithm
	// requires:  x<modulus,  modulus is odd

	// compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
	cgbn_negate(_env, t, modulus);
	cgbn_store(_env, window+0, t);

	// convert x into Montgomery space, store into window table
	np0=cgbn_bn2mont(_env, result, x, modulus);
	cgbn_store(_env, window+1, result);
	cgbn_set(_env, t, result);

	// compute x^2, x^3, ... x^(2^window_bits-1), store into window table
	for(index=2;index<(1<<window_bits);index++) {
		cgbn_mont_mul(_env, result, result, t, modulus, np0);
		cgbn_store(_env, window+index, result);
	}

	// find leading high bit
	position=BITS - cgbn_clz(_env, power);
	// break the exponent into chunks, each window_bits in length
	// load the most significant non-zero exponent chunk
	offset=position % window_bits;
	if(offset==0)
		position=position-window_bits;
	else
		position=position-offset;
	index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
	cgbn_load(_env, result, window+index);

	// process the remaining exponent chunks
	while(position>0) {
		// square the result window_bits times
		for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
			cgbn_mont_sqr(_env, result, result, modulus, np0);

		// multiply by next exponent chunk
		position=position-window_bits;
		index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
		cgbn_load(_env, t, window+index);
		cgbn_mont_mul(_env, result, result, t, modulus, np0);
	}

	// we've processed the exponent now, convert back to normal space
	cgbn_mont2bn(_env, result, result, modulus, np0);
}


__device__ __forceinline__ void sliding_window_powm_odd(env_t _env, env_t::cgbn_t &result, env_t::cgbn_t &x, env_t::cgbn_t &power, env_t::cgbn_t &modulus) {
	env_t::cgbn_t         t, starts;
	int32_t               index, position, leading;
	uint32_t              mont_inv;
	env_t::cgbn_local_t   odd_powers[1<<window_bits-1];

	// conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
	// requires:  x<modulus,  modulus is odd

	// find the leading one in the power
	leading=BITS-1-cgbn_clz(_env, power);
	if(leading>=0) {
		// convert x into Montgomery space, store in the odd powers table
		mont_inv=cgbn_bn2mont(_env, result, x, modulus);

		// compute t=x^2 mod modulus
		cgbn_mont_sqr(_env, t, result, modulus, mont_inv);

		// compute odd powers window table: x^1, x^3, x^5, ...
		cgbn_store(_env, odd_powers, result);
		for(index=1;index<(1<<window_bits-1);index++) {
			cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
			cgbn_store(_env, odd_powers+index, result);
		}

		// starts contains an array of bits indicating the start of a window
		cgbn_set_ui32(_env, starts, 0);
		// organize p as a sequence of odd window indexes
		position=0;
		while(true) {
			if(cgbn_extract_bits_ui32(_env, power, position, 1)==0)
				position++;
			else {
				cgbn_insert_bits_ui32(_env, starts, starts, position, 1, 1);
				if(position+window_bits>leading)
					break;
				position=position+window_bits;
			}
		}

		// load first window.  Note, since the window index must be odd, we have to
		// divide it by two before indexing the window table.  Instead, we just don't
		// load the index LSB from power
		index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
		cgbn_load(_env, result, odd_powers+index);
		position--;

		// Process remaining windows 
		while(position>=0) {
			cgbn_mont_sqr(_env, result, result, modulus, mont_inv);
			if(cgbn_extract_bits_ui32(_env, starts, position, 1)==1) {
				// found a window, load the index
				index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
				cgbn_load(_env, t, odd_powers+index);
				cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
			}
			position--;
		}
		// convert result from Montgomery space
		cgbn_mont2bn(_env, result, result, modulus, mont_inv);
	}
	else {
		// p=0, thus x^p mod modulus=1
		cgbn_set_ui32(_env, result, 1);
	}
}


__global__ void kernel_powm_fixed_window(instance_t *instances, uint32_t count) {
	int32_t instance;

	// decode an instance number from the blockIdx and threadIdx
	instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
	if(instance>=count)
		return;

	context_t      bn_context(cgbn_no_checks, NULL, instance);   // construct a context
	env_t          bn_env(bn_context.env<env_t>());              // construct an environment for 4096-bit math
	env_t::cgbn_t  r, x, p, m;                                   // define r, x, e, m as 4096-bit bignums

	cgbn_load(bn_env, x, &(instances[instance].x));            // load my instance's x value
	cgbn_load(bn_env, p, &(instances[instance].power));        // load my instance's p value
	cgbn_load(bn_env, m, &(instances[instance].modulus));      // load my instance's m value
	fixed_window_powm_odd(bn_env, r, x, p, m);                 // r = x^p % m
	cgbn_store(bn_env, &(instances[instance].result), r);      // store r into sum
}


__global__ void kernel_powm_sliding_window(instance_t *instances, uint32_t count) {
	int32_t instance;

	// decode an instance number from the blockIdx and threadIdx
	instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
	if(instance>=count)
		return;

	context_t      bn_context(cgbn_no_checks, NULL, instance);   // construct a context
	env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 4096-bit math
	env_t::cgbn_t  r, x, p, m;                                          // define r, x, e, m as 4096-bit bignums

	cgbn_load(bn_env, x, &(instances[instance].x));            // load my instance's x value
	cgbn_load(bn_env, p, &(instances[instance].power));        // load my instance's p value
	cgbn_load(bn_env, m, &(instances[instance].modulus));      // load my instance's m value
	sliding_window_powm_odd(bn_env, r, x, p, m);                 // r = x^p % m
	cgbn_store(bn_env, &(instances[instance].result), r);      // store r into sum
}


// the actual kernel
__global__ void kernel_modular_power(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
	int32_t instance;

	// decode an instance number from the blockIdx and threadIdx
	instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
	if(instance>=count)
		return;

	//context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
	context_t      bn_context(cgbn_no_checks, NULL, instance);   // construct a context
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


extern "C" void powm_fixed_window(int size) {
	instance_t          *instances, *gpuInstances;
	int32_t TPB = 128;
	int32_t IPB = TPB / TPI;

	Instances = size;
	instances = (instance_t *)malloc(sizeof(instance_t)*Instances);
	generate_random_instances(instances, Instances);

	cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*Instances);
	cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*Instances, cudaMemcpyHostToDevice);

	kernel_powm_fixed_window<<<(Instances + IPB - 1)/IPB, TPB>>>(gpuInstances, Instances);
	cudaDeviceSynchronize();

	cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*Instances, cudaMemcpyDeviceToHost);
	//verify_results(instances, Instances);

	// clean up
	free(instances);
	cudaFree(gpuInstances);
}


extern "C" void tss_powm_fixed_window(int size) {
    instance_t          instances, *gpuInstances;
    unsigned long x[BITS/32], y[BITS/32], m[BITS/32], z[BITS/32];
    int32_t TPB = 128;
    int32_t IPB = TPB / TPI;

    for (int index = 0; index < BITS/32; index++) {
        instances.x._limbs[index] = (uint32_t)x[index];
        instances.power._limbs[index] = (uint32_t)y[index];
        instances.modulus._limbs[index] = (uint32_t)m[index];
        instances.result._limbs[index] = 0;
    }
    generate_random_instances(&instances, 1);

    cudaMalloc((void **)&gpuInstances, sizeof(instance_t));
    cudaMemcpy(gpuInstances, &instances, sizeof(instance_t), cudaMemcpyHostToDevice);

    kernel_powm_fixed_window<<<(1 + IPB - 1)/IPB, TPB>>>(gpuInstances, 1);
    //cudaDeviceSynchronize();

    cudaMemcpy(&instances, gpuInstances, sizeof(instance_t), cudaMemcpyDeviceToHost);
    verify_results(&instances, Instances);

    for (int index = 0; index < BITS/32; index++)
        z[index] = (unsigned long)instances.result._limbs[index];

    // clean up
    cudaFree(gpuInstances);
}


extern "C" void powm_sliding_window(int size) {
	instance_t          *instances, *gpuInstances;
	int32_t TPB = 128;
	int32_t IPB = TPB / TPI;

	Instances = size;
	instances = (instance_t *)malloc(sizeof(instance_t)*Instances);
	generate_random_instances(instances, Instances);

	cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*Instances);
	cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*Instances, cudaMemcpyHostToDevice);

	kernel_powm_sliding_window<<<(Instances + IPB - 1)/IPB, TPB>>>(gpuInstances, Instances);
	cudaDeviceSynchronize();

	cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*Instances, cudaMemcpyDeviceToHost);
	//verify_results(instances, Instances);

	// clean up
	free(instances);
	cudaFree(gpuInstances);
}


extern "C" void cgbn_power(int size) {
	instance_t          *instances, *gpuInstances;
	cgbn_error_report_t *report;
	int32_t TPB = 128;
	int32_t IPB = TPB / TPI;

	Instances = size;
	instances = (instance_t *)malloc(sizeof(instance_t)*Instances);
	generate_random_instances(instances, Instances);
	cgbn_error_report_alloc(&report);

	cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*Instances);
	cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*Instances, cudaMemcpyHostToDevice);

	kernel_modular_power<<<(Instances + IPB - 1)/IPB, TPB>>>(report, gpuInstances, Instances);
	cudaDeviceSynchronize();

	cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*Instances, cudaMemcpyDeviceToHost);
	verify_results(instances, Instances);

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
