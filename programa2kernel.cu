/*
152096 - William Matheus
Friendly Numbers
Programacao Paralela e Distribuida
CUDA - 2019/2 - UPF
Programa 2 - Kernel
*/


#include <stdio.h>
#include <cuda.h>

__device__ void divisionRemainder(int num, int den, int *divisionRemainder){
	*divisionRemainder = num / den;
	*divisionRemainder = num - (den * *divisionRemainder);

}

__device__ void gcd ( int a, int b, int *result){
	int c, resto;	
	while ( a != 0 ) {
     	c = a; 
     	divisionRemainder(b, a, &resto);
     	a = resto;
		b = c;
	}
    *result = b;
}


__global__ void numDem(long int *device_num, long int *device_den, long int start, long int end, int size, int inc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + inc;
	int result, resto;
	long int factor, ii, sum, done, n;

	if (i < size) {
		ii = i - start;
		sum = 1 + i;
		done = i;
		factor = 2;
		while (factor < done) {
			divisionRemainder(i, factor, &resto);
			if (resto== 0) {
				sum += (factor + (i / factor));
				//doneFactor(i, factor, &done);
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

__global__ void sum(long int* device_num, long int* device_den, long int* device_vet, int size, int x)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + x;
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


	int deviceId;
	int number_grid, c=0, i;
	long int *device_num, *device_den, *device_vet;

	struct cudaDeviceProp prop;

	cudaGetDevice(&deviceId);
	cudaGetDeviceProperties(&prop, deviceId);


	long int last = end - start + 1;
	size_t size = last * sizeof(long int);

	int nBlocks = 4*((prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount)/prop.maxThreadsPerBlock);	
	int threads = prop.maxThreadsPerBlock;

	number_grid = last / (nBlocks * threads)+1;

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

	cudaMemcpy(device_num, num, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_den, den, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_vet, vet, size, cudaMemcpyHostToDevice);

	int x = 0;

	for (i = 0; i < number_grid; i++) {
		numDem<<<nBlocks, threads>>>(device_num, device_den, start, end, last, x);
		x += nBlocks * threads;
	}
	
	x = 0;

	for (i = 0; i < number_grid; i++) {
		sum<<<nBlocks, threads>>>(device_num, device_den, device_vet, last, x);
		x += nBlocks * threads;;
	}

	cudaMemcpy(vet, device_vet, size, cudaMemcpyDeviceToHost);

	for (i = 0; i < last; i++) {
		c += vet[i];
	}

	printf("Found %d pairs of mutually friendly numbers\n", c);

	free(num);
	free(den);
	free(vet);

	cudaFree(device_num);
	cudaFree(device_den);
	cudaFree(device_vet);
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
