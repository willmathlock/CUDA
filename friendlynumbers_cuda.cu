/*
152096 - William Matheus
Friendly Numbers
Programacao Paralela e Distribuida
CUDA - 2019/2 - UPF
*/


#include <stdio.h>
#include <cuda.h>


#define THREADSPERBLOCK 1024
#define NUMBEROFBLOCK 	3

__device__ void gcd ( int a, int b, int *result){
	int c;	
	while ( a != 0 ) {
     	c = a; 
     	a = b % a;
		b = c;
	}
    *result = b;
}

__global__ void FuncNumDem(long int* device_num, long int* device_den, long int start, long int end, int size, int inc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + inc;
	int result;
	long int factor, ii, sum, done, n;

	if (i < size) {
		ii = i - start;
		sum = 1 + i;
		done = i;
		factor = 2;
		while (factor < done) {
			if ((i % factor) == 0) {
				sum += (factor + (i / factor));
				if ((done = i / factor) == factor)
					sum -= factor;
			}
			factor++;
		}
		device_num[ii] = sum;
		device_den[ii] = i;
	 	gcd(device_num[ii], device_den[ii], &result);
		n = result;
		device_num[ii] /= n;
		device_den[ii] /= n;
	}
}

__global__ void FuncSum(long int* device_num, long int* device_den, long int* device_vet, int size, int incremento)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + incremento;
	int j;

	if (i < size) {
		for (j = i + 1; j < size; j++) {
			if ((device_num[i] == device_num[j]) && (device_den[i] == device_den[j]))
				device_vet[i]++;
		}
	}
}

void friendly_numbers(long int start, long int end) {
	cudaSetDevice(0);

	int number_grid, c=0, i;
	long int *device_num, *device_den, *device_vet;

	long int last = end - start + 1;
	size_t size = last * sizeof(long int);
	number_grid = last / (NUMBEROFBLOCK * THREADSPERBLOCK)+1;

	long int *num;
	long int *den;
	long int *vet;

	num = (long int*) malloc(size);
	den = (long int*) malloc(size);
	vet = (long int*) malloc(size);

	cudaMalloc((void**)&device_num, size);
	cudaMalloc((void**)&device_den, size);
	cudaMalloc((void**)&device_vet, size);
	
	for (i = 0; i < last; i++) {
		vet[i] = 0;
	}

	cudaMemcpy(cuda_num, num, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_den, den, size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_vet, vet, size, cudaMemcpyHostToDevice);

	int incremento = 0;

	for (i = 0; i < number_grid; i++) {
		FuncNumDem<<<NUMBEROFBLOCK, THREADSPERBLOCK>>>(cuda_num, cuda_den, start, end, last, incremento);
		incremento += NUMBEROFBLOCK * THREADSPERBLOCK;
	}
	
	incremento = 0;

	for (i = 0; i < number_grid; i++) {
		FuncSum<<<NUMBEROFBLOCK, THREADSPERBLOCK>>>(cuda_num, cuda_den, cuda_vet, last, incremento);
		incremento += NUMBEROFBLOCK * THREADSPERBLOCK;
	}

	cudaMemcpy(vet, cuda_vet, size, cudaMemcpyDeviceToHost);

	for (i = 0; i < last; i++) {
		c += vet[i];
	}

	printf("Found %d pairs of mutually friendly numbers\n", c);

	free(num);
	free(den);
	free(vet);

	cudaFree(cuda_num);
	cudaFree(cuda_den);
	cudaFree(cuda_vet);
}

int main(int argc, char **argv) {
	long int start;
	long int end;

	if (argc != 3){
		printf("Wrong number of arguments\n");
		return EXIT_FAILURE;
	}

	start = atoi(argv[1]);
	end = atoi(argv[2]);

	printf("Number %ld to %ld\n", start, end);
	friendly_numbers(start, end);

	return EXIT_SUCCESS;
}
