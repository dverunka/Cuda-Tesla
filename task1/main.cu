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

#define VECTOR_SIZE (1234u)
#define BLOCK_SIZE (16u)

// Fill vector with numbers.
void init_vector(int * data) {

    for (unsigned int i = 0; i < VECTOR_SIZE; i++) {
        data[i] = i + 1;
    }
}

// Print first 5 integers.
void print_vector(int * data) {

    for (int i = 0; i < 5; i++) {
        printf("%d, ", data[i]);
    }
    printf("\n");
}


// Add 2 vectors.
__global__ void add_vectors(int * vector1, int * vector2, int * result_vector) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < VECTOR_SIZE) {
        result_vector[i] = vector1[i] + vector2[i];
    }
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {

    int grid_size = (VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Init vectors on host.
    int *host_vector1 = (int*) malloc(VECTOR_SIZE * sizeof(int));
    init_vector(host_vector1);

    int *host_vector2 = (int*) malloc(VECTOR_SIZE * sizeof(int));
    init_vector(host_vector2);

    int *host_vector3 = (int*) malloc(VECTOR_SIZE * sizeof(int));

    // Init vectors on graphic card.
    int *card_vector1, *card_vector2, *card_vector3 = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&card_vector1, VECTOR_SIZE * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&card_vector2, VECTOR_SIZE * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&card_vector3, VECTOR_SIZE * sizeof(int)));

    // Copy host vectors to graphic card.
    CUDA_CHECK_RETURN(cudaMemcpy(card_vector1, host_vector1, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(card_vector2, host_vector2, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    add_vectors <<< grid_size, BLOCK_SIZE >>> (card_vector1, card_vector2, card_vector3); // ignore warning

    // Rather wait.
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Copy result vector back to host.
    CUDA_CHECK_RETURN(cudaMemcpy(host_vector3, card_vector3, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    print_vector(host_vector3);

    // Clean memory.
    free(host_vector1);
    free(host_vector2);
    free(host_vector3);

    CUDA_CHECK_RETURN(cudaFree(card_vector1));
    CUDA_CHECK_RETURN(cudaFree(card_vector2));
    CUDA_CHECK_RETURN(cudaFree(card_vector3));
	
	return 0;
}
