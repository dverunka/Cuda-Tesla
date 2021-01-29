/**
 * Nasobeni matic v globalni pameti.
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

typedef struct {
    int width;
    int height;
    float* values;
} Matrix;

#define BLOCK_SIZE (3)
#define MATRIX1_WIDTH (2u)
#define MATRIX1_HEIGHT (3u)
#define MATRIX2_WIDTH (3u)
#define MATRIX2_HEIGHT (2u)

// Fill matrix with numbers.
void init_matrix(Matrix * data) {

    for (int i = 0; i < data->width * data->height; i++) {
        data->values[i] = i + 1;
    }
}

// Print matrix.
void print_matrix(Matrix * data) {

    for (int i = 0; i < data->width * data->height; i++) {

        printf("%f ", data->values[i]);

        if ((i + 1) % data->width == 0) {
            printf("\n");
        }
    }
}

// Multiply 2 matrices.
__global__ void multiply_matrices(Matrix * matrix1, Matrix * matrix2, Matrix * matrix3) {

    float value = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    for (unsigned int i = 0; i < matrix1->width; i++) {
        value += matrix1->values[row * matrix1->width + i] * matrix2->values[i * matrix2->width + column];
    }
    matrix3->values[row * matrix3->width + column] = value;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {

    // Init matrices on host.
    Matrix *host_matrix1 = (Matrix*) malloc(sizeof(Matrix));
    Matrix *host_matrix2 = (Matrix*) malloc(sizeof(Matrix));
    Matrix *host_matrix3 = (Matrix*) malloc(sizeof(Matrix));

    host_matrix1->width = MATRIX1_WIDTH;
    host_matrix1->height = MATRIX1_HEIGHT;
    host_matrix2->width = MATRIX2_WIDTH;
    host_matrix2->height = MATRIX2_HEIGHT;
    host_matrix3->width = MATRIX2_WIDTH; // width from matrix 2
    host_matrix3->height = MATRIX1_HEIGHT; // height from matrix 1

    host_matrix1->values = (float*) malloc(MATRIX1_WIDTH * MATRIX1_HEIGHT * sizeof(float));
    host_matrix2->values = (float*) malloc(MATRIX2_WIDTH * MATRIX2_HEIGHT * sizeof(float));
    host_matrix3->values = (float*) malloc(MATRIX2_WIDTH * MATRIX1_HEIGHT * sizeof(float));
    
    init_matrix(host_matrix1);
    init_matrix(host_matrix2);

    print_matrix(host_matrix1);
    printf("\n");
    print_matrix(host_matrix2);
    printf("\n");

    // Init matrices on graphic card.
    Matrix *card_matrix1, *card_matrix2, *card_matrix3 = NULL;
    float *card_matrix1_values, *card_matrix2_values, *card_matrix3_values;

    CUDA_CHECK_RETURN(cudaMalloc(&card_matrix1, sizeof(Matrix)));
    CUDA_CHECK_RETURN(cudaMalloc(&card_matrix2, sizeof(Matrix)));
    CUDA_CHECK_RETURN(cudaMalloc(&card_matrix3, sizeof(Matrix)));

    CUDA_CHECK_RETURN(cudaMalloc(&card_matrix1_values, MATRIX1_WIDTH * MATRIX1_HEIGHT * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc(&card_matrix2_values, MATRIX2_WIDTH * MATRIX2_HEIGHT * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc(&card_matrix3_values, MATRIX2_WIDTH * MATRIX1_HEIGHT * sizeof(float)));

    int w1 = MATRIX1_WIDTH; int w2 = MATRIX2_WIDTH;
    int h1 = MATRIX1_HEIGHT; int h2 = MATRIX2_HEIGHT;

    // Copy host matrices to graphic card.
    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix1->width), &w1, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix1->height), &h1, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(card_matrix1_values, host_matrix1->values, MATRIX1_WIDTH * MATRIX1_HEIGHT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix1->values), &card_matrix1_values, sizeof(float*), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix2->width), &w2, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix2->height), &h2, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(card_matrix2_values, host_matrix2->values, MATRIX2_WIDTH * MATRIX2_HEIGHT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix2->values), &card_matrix2_values, sizeof(float*), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix3->width), &w2, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix3->height), &h1, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(&(card_matrix3->values), &card_matrix3_values, sizeof(float*), cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(host_matrix3->width / dimBlock.x, host_matrix3->height / dimBlock.y);

    multiply_matrices <<< dimGrid, dimBlock >>> (card_matrix1, card_matrix2, card_matrix3); // ignore warning

    // Rather wait.
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Copy result matrix back to host.
    CUDA_CHECK_RETURN(cudaMemcpy(host_matrix3->values, card_matrix3_values, MATRIX2_WIDTH * MATRIX1_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost));

    print_matrix(host_matrix3);

    // Clear memory.
    free(host_matrix1->values);
    free(host_matrix2->values);
    free(host_matrix3->values);

    free(host_matrix1);
    free(host_matrix2);
    free(host_matrix3);

    CUDA_CHECK_RETURN(cudaFree(card_matrix1));
    CUDA_CHECK_RETURN(cudaFree(card_matrix2));
    CUDA_CHECK_RETURN(cudaFree(card_matrix3));

    CUDA_CHECK_RETURN(cudaFree(card_matrix1_values));
    CUDA_CHECK_RETURN(cudaFree(card_matrix2_values));
    CUDA_CHECK_RETURN(cudaFree(card_matrix3_values));
	
	return 0;
}
