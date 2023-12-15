#include <stdio.h>
//#include <ctype.h>
//#include <stdint.h>
//#include <stdlib.h>
//#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gmp.h>

#include "cgbn/cgbn.h"
#include "../samples/utility/cpu_support.h"
#include "../samples/utility/cpu_simple_bn_math.h"
#include "../samples/utility/gpu_support.h"


// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define window_bits 7

class params_t {
  public:
    // parameters used by the CGBN context
    static const uint32_t TPB=0;                     // get TPB from blockDim.x
    static const uint32_t MAX_ROTATION=4;            // good default value
    static const uint32_t SHM_LIMIT=0;               // no shared mem available
    static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet
};

typedef params_t params;

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI, params> context_t;
typedef cgbn_env_t<context_t, 4096> env_4096_t;
typedef cgbn_env_t<context_t, 3072> env_3072_t;
typedef cgbn_env_t<context_t, 2048> env_2048_t;

uint32_t Instances=10;

// Declare the instance type
template<uint32_t bits>
struct instance_t{
    cgbn_mem_t<bits> x;
    cgbn_mem_t<bits> power;
    cgbn_mem_t<bits> modulus;
    cgbn_mem_t<bits> result;
};

typedef instance_t<4096> instance_4096_t;
typedef instance_t<3072> instance_3072_t;
typedef instance_t<2048> instance_2048_t;


void myPrint(uint32_t *x, uint32_t count) {
	int32_t index;

	for(index=count-1;index>=0;index--)
		printf("%u ", x[index]);
	printf("\n");
}


// gnerate random input
void generate_random_instances_4096(instance_4096_t *instances, uint32_t count, uint32_t bits) {
	int         index;

	for(index=0;index<count;index++) {
		random_words(instances[index].x._limbs, bits/32);
		random_words(instances[index].power._limbs, bits/32);
		random_words(instances[index].modulus._limbs, bits/32);
		zero_words(instances[index].result._limbs, bits/32);

		// ensure modulus is odd
		instances[index].modulus._limbs[0] |= 1;

		// ensure modulus is greater than 
		if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32)>0) {
			swap_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32);

			// modulus might now be even, ensure it's odd
			instances[index].modulus._limbs[0] |= 1;
		}
		else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32)==0) {
			// since modulus is odd and modulus = x, we can just subtract 1 from x
			instances[index].x._limbs[0] -= 1;
		}
	}
}

void generate_random_instances_3072(instance_3072_t *instances, uint32_t count, uint32_t bits) {
	int         index;

	for(index=0;index<count;index++) {
		random_words(instances[index].x._limbs, bits/32);
		random_words(instances[index].power._limbs, bits/32);
		random_words(instances[index].modulus._limbs, bits/32);
		zero_words(instances[index].result._limbs, bits/32);

		// ensure modulus is odd
		instances[index].modulus._limbs[0] |= 1;

		// ensure modulus is greater than 
		if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32)>0) {
			swap_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32);

			// modulus might now be even, ensure it's odd
			instances[index].modulus._limbs[0] |= 1;
		}
		else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32)==0) {
			// since modulus is odd and modulus = x, we can just subtract 1 from x
			instances[index].x._limbs[0] -= 1;
		}
	}
}


void generate_random_instances_2048(instance_2048_t *instances, uint32_t count, uint32_t bits) {
	int         index;

	for(index=0;index<count;index++) {
		random_words(instances[index].x._limbs, bits/32);
		random_words(instances[index].power._limbs, bits/32);
		random_words(instances[index].modulus._limbs, bits/32);
		zero_words(instances[index].result._limbs, bits/32);

		// ensure modulus is odd
		instances[index].modulus._limbs[0] |= 1;

		// ensure modulus is greater than 
		if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32)>0) {
			swap_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32);

			// modulus might now be even, ensure it's odd
			instances[index].modulus._limbs[0] |= 1;
		}
		else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, bits/32)==0) {
			// since modulus is odd and modulus = x, we can just subtract 1 from x
			instances[index].x._limbs[0] -= 1;
		}
	}
}


// support routine to verify the GPU results using the CPU
void verify_results_4096(instance_4096_t *instances, uint32_t count, uint32_t bits) {
	mpz_t x, p, m, computed, correct;

	mpz_init(x);
	mpz_init(p);
	mpz_init(m);
	mpz_init(computed);
	mpz_init(correct);

	uint32_t matched = 0;
	for(uint32_t index=0;index<count;index++) {
		//printf("[%u]\n", index);
		to_mpz(x, instances[index].x._limbs, bits/32);
		to_mpz(p, instances[index].power._limbs, bits/32);
		to_mpz(m, instances[index].modulus._limbs, bits/32);
		to_mpz(computed, instances[index].result._limbs, bits/32);

		mpz_powm(correct, x, p, m);

		uint32_t c[bits/32] = {0};
		from_mpz(correct, c, bits/32);

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

void verify_results_3072(instance_3072_t *instances, uint32_t count, uint32_t bits) {
	mpz_t x, p, m, computed, correct;

	mpz_init(x);
	mpz_init(p);
	mpz_init(m);
	mpz_init(computed);
	mpz_init(correct);

	uint32_t matched = 0;
	for(uint32_t index=0;index<count;index++) {
		//printf("[%u]\n", index);
		to_mpz(x, instances[index].x._limbs, bits/32);
		to_mpz(p, instances[index].power._limbs, bits/32);
		to_mpz(m, instances[index].modulus._limbs, bits/32);
		to_mpz(computed, instances[index].result._limbs, bits/32);

		mpz_powm(correct, x, p, m);

		uint32_t c[bits/32] = {0};
		from_mpz(correct, c, bits/32);

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

void verify_results_2048(instance_2048_t *instances, uint32_t count, uint32_t bits) {
	mpz_t x, p, m, computed, correct;

	mpz_init(x);
	mpz_init(p);
	mpz_init(m);
	mpz_init(computed);
	mpz_init(correct);

	uint32_t matched = 0;
	for(uint32_t index=0;index<count;index++) {
		//printf("[%u]\n", index);
		to_mpz(x, instances[index].x._limbs, bits/32);
		to_mpz(p, instances[index].power._limbs, bits/32);
		to_mpz(m, instances[index].modulus._limbs, bits/32);
		to_mpz(computed, instances[index].result._limbs, bits/32);

		mpz_powm(correct, x, p, m);

		uint32_t c[bits/32] = {0};
		from_mpz(correct, c, bits/32);

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

/***************************************/

__device__ __forceinline__ void sliding_window_powm_4096(env_4096_t _env, env_4096_t::cgbn_t &result, env_4096_t::cgbn_t &x, env_4096_t::cgbn_t &power, env_4096_t::cgbn_t &modulus, uint32_t bits) {
	env_4096_t::cgbn_t         t, starts;
	int32_t               index, position, leading;
	uint32_t              mont_inv;
	env_4096_t::cgbn_local_t   odd_powers[1<<window_bits-1];

	// conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
	// requires:  x<modulus,  modulus is odd

	// find the leading one in the power
	leading=bits-1-cgbn_clz(_env, power);
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


__device__ __forceinline__ void sliding_window_powm_3072(env_3072_t _env, env_3072_t::cgbn_t &result, env_3072_t::cgbn_t &x, env_3072_t::cgbn_t &power, env_3072_t::cgbn_t &modulus, uint32_t bits) {
	env_3072_t::cgbn_t         t, starts;
	int32_t               index, position, leading;
	uint32_t              mont_inv;
	env_3072_t::cgbn_local_t   odd_powers[1<<window_bits-1];

	// conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
	// requires:  x<modulus,  modulus is odd

	// find the leading one in the power
	leading=bits-1-cgbn_clz(_env, power);
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

__device__ __forceinline__ void sliding_window_powm_2048(env_2048_t _env, env_2048_t::cgbn_t &result, env_2048_t::cgbn_t &x, env_2048_t::cgbn_t &power, env_2048_t::cgbn_t &modulus, uint32_t bits) {
	env_2048_t::cgbn_t         t, starts;
	int32_t               index, position, leading;
	uint32_t              mont_inv;
	env_2048_t::cgbn_local_t   odd_powers[1<<window_bits-1];

	// conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
	// requires:  x<modulus,  modulus is odd

	// find the leading one in the power
	leading=bits-1-cgbn_clz(_env, power);
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


/***************************************/

__global__ void kernel_powm_sliding_window_4096(instance_4096_t *instances, uint32_t count) {
	int32_t instance;

	// decode an instance number from the blockIdx and threadIdx
	instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
	if(instance>=count)
		return;

	context_t      bn_context(cgbn_no_checks, NULL, instance);   // construct a context
	env_4096_t          bn_env(bn_context.env<env_4096_t>());                     // construct an environment for 4096-bit math
	env_4096_t::cgbn_t  r, x, p, m;                                          // define r, x, e, m as 4096-bit bignums

	cgbn_load(bn_env, x, &(instances[instance].x));            // load my instance's x value
	cgbn_load(bn_env, p, &(instances[instance].power));        // load my instance's p value
	cgbn_load(bn_env, m, &(instances[instance].modulus));      // load my instance's m value
	sliding_window_powm_4096(bn_env, r, x, p, m, 4096);                 // r = x^p % m
	cgbn_store(bn_env, &(instances[instance].result), r);      // store r into sum
}


__global__ void kernel_powm_sliding_window_3072(instance_3072_t *instances, uint32_t count) {
	int32_t instance;

	// decode an instance number from the blockIdx and threadIdx
	instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
	if(instance>=count)
		return;

	context_t      bn_context(cgbn_no_checks, NULL, instance);   // construct a context
	env_3072_t          bn_env(bn_context.env<env_3072_t>());    // construct an environment for 4096-bit math
	env_3072_t::cgbn_t  r, x, p, m;                              // define r, x, e, m as 4096-bit bignums

	cgbn_load(bn_env, x, &(instances[instance].x));            // load my instance's x value
	cgbn_load(bn_env, p, &(instances[instance].power));        // load my instance's p value
	cgbn_load(bn_env, m, &(instances[instance].modulus));      // load my instance's m value
	sliding_window_powm_3072(bn_env, r, x, p, m, 3072);                 // r = x^p % m
	cgbn_store(bn_env, &(instances[instance].result), r);      // store r into sum
}


__global__ void kernel_powm_sliding_window_2048(instance_2048_t *instances, uint32_t count) {
	int32_t instance;

	// decode an instance number from the blockIdx and threadIdx
	instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
	if(instance>=count)
		return;

	context_t      bn_context(cgbn_no_checks, NULL, instance);   // construct a context
	env_2048_t          bn_env(bn_context.env<env_2048_t>());    // construct an environment for 4096-bit math
	env_2048_t::cgbn_t  r, x, p, m;                              // define r, x, e, m as 4096-bit bignums

	cgbn_load(bn_env, x, &(instances[instance].x));            // load my instance's x value
	cgbn_load(bn_env, p, &(instances[instance].power));        // load my instance's p value
	cgbn_load(bn_env, m, &(instances[instance].modulus));      // load my instance's m value
	sliding_window_powm_2048(bn_env, r, x, p, m, 2048);                 // r = x^p % m
	cgbn_store(bn_env, &(instances[instance].result), r);      // store r into sum
}

/***************************************/

extern "C" void powm_sliding_window(int size) {
    /*
	instance_4096_t *instances, *gpuInstances;
	int32_t TPB = 128;
	int32_t IPB = TPB / TPI;

	Instances = size;
	instances = (instance_4096_t *)malloc(sizeof(instance_4096_t)*Instances);
	generate_random_instances(instances, Instances);

	cudaMalloc((void **)&gpuInstances, sizeof(instance_4096_t)*Instances);
	cudaMemcpy(gpuInstances, instances, sizeof(instance_4096_t)*Instances, cudaMemcpyHostToDevice);

	kernel_powm_sliding_window<<<(Instances + IPB - 1)/IPB, TPB>>>(gpuInstances, Instances);
	cudaDeviceSynchronize();

	cudaMemcpy(instances, gpuInstances, sizeof(instance_4096_t)*Instances, cudaMemcpyDeviceToHost);
	//verify_results(instances, Instances);

	// clean up
	free(instances);
	cudaFree(gpuInstances);
    */
}


int main() {
	int size = 100;
    /*
    instance_4096_t  *instances, *gpuInstances;
    int32_t TPB = 128;
    int32_t IPB = TPB / TPI;

	printf("Genereating instances: %u ...\n", size);
    instances = (instance_4096_t *)malloc(sizeof(instance_4096_t)*size);
    generate_random_instances_4096(instances, size, 4096);

	printf("Copying instances to the GPU ...\n");
    cudaMalloc((void **)&gpuInstances, sizeof(instance_4096_t)*size);
    cudaMemcpy(gpuInstances, instances, sizeof(instance_4096_t)*size, cudaMemcpyHostToDevice);

	printf("Running GPU kernel ...\n");
    kernel_powm_sliding_window_4096<<<(size + IPB - 1)/IPB, TPB>>>(gpuInstances, size);
    cudaDeviceSynchronize();

	printf("Copying results back to CPU ...\n");
    cudaMemcpy(instances, gpuInstances, sizeof(instance_4096_t)*size, cudaMemcpyDeviceToHost);

	//printf("Verifying the results ...\n");
    verify_results_4096(instances, size, 4096);

    // clean up
    free(instances);
    cudaFree(gpuInstances);

    instance_3072_t  *instances, *gpuInstances;
    int32_t TPB = 128;
    int32_t IPB = TPB / TPI;

	printf("Genereating instances: %u ...\n", size);
    instances = (instance_3072_t *)malloc(sizeof(instance_3072_t)*size);
    generate_random_instances_3072(instances, size, 3072);

	printf("Copying instances to the GPU ...\n");
    cudaMalloc((void **)&gpuInstances, sizeof(instance_3072_t)*size);
    cudaMemcpy(gpuInstances, instances, sizeof(instance_3072_t)*size, cudaMemcpyHostToDevice);

	printf("Running GPU kernel ...\n");
    kernel_powm_sliding_window_3072<<<(size + IPB - 1)/IPB, TPB>>>(gpuInstances, size);
    cudaDeviceSynchronize();

	printf("Copying results back to CPU ...\n");
    cudaMemcpy(instances, gpuInstances, sizeof(instance_3072_t)*size, cudaMemcpyDeviceToHost);

	//printf("Verifying the results ...\n");
    verify_results_3072(instances, size, 3072);

    // clean up
    free(instances);
    cudaFree(gpuInstances);
*/

    instance_2048_t  *instances, *gpuInstances;
    int32_t TPB = 128;
    int32_t IPB = TPB / TPI;

	printf("Genereating instances: %u ...\n", size);
    instances = (instance_2048_t *)malloc(sizeof(instance_2048_t)*size);
    generate_random_instances_2048(instances, size, 2048);

	printf("Copying instances to the GPU ...\n");
    cudaMalloc((void **)&gpuInstances, sizeof(instance_2048_t)*size);
    cudaMemcpy(gpuInstances, instances, sizeof(instance_2048_t)*size, cudaMemcpyHostToDevice);

	printf("Running GPU kernel ...\n");
    kernel_powm_sliding_window_2048<<<(size + IPB - 1)/IPB, TPB>>>(gpuInstances, size);
    cudaDeviceSynchronize();

	printf("Copying results back to CPU ...\n");
    cudaMemcpy(instances, gpuInstances, sizeof(instance_2048_t)*size, cudaMemcpyDeviceToHost);

	//printf("Verifying the results ...\n");
    verify_results_2048(instances, size, 2048);

    // clean up
    free(instances);
    cudaFree(gpuInstances);

	return 0;
}
