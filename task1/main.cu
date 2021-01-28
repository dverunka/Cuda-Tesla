/**
 * Soucet vektoru pomoci CUDA API.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {                                                                          \
                                                                                                            \
	cudaError_t err = value;                                                                                \
                                                                                                            \
	if (err != cudaSuccess) {                                                                               \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__);   \
		exit(1);                                                                                            \
	}                                                                                                       \
}                                                                                                           \

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {


	
	return 0;
}
