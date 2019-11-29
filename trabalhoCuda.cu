#include <stdio.h>
#include <cuda.h>

__device__ void restoDivisao(int num, int den, int *resto)
{
	*resto = num / den;
	*resto = num - (den * *resto);
}

__device__ void gcd ( int a, int b, int *ret){
  	int c, resto;
	while ( a != 0 ) {
     	c = a; 
     	restoDivisao(b, a, &resto);
		a = resto;  
		b = c;
	}

    *ret = b;
}

__global__ void NumDemSearch(long int* d_the_num, long int* d_num, long int* d_den, long int start, long int end, int size, int inc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + inc;
	int resto, ret;
	long int factor, ii, sum, done, n;

	if (i < size) {
		ii = i - start;
		sum = 1 + i;
		d_the_num[ii] = i;
		done = i;
		factor = 2;
		while (factor < done) {
			restoDivisao(i, factor, &resto);

			if (resto == 0) {
				sum += (factor + (i / factor));
				if ((done = i / factor) == factor)
					sum -= factor;
			}
			factor++;
		}
		d_num[ii] = sum;
		d_den[ii] = i;
	 	gcd(d_num[ii], d_den[ii], &ret);
		n = ret;
		d_num[ii] /= n;
		d_den[ii] /= n;
	}
}

__global__ void SumFriendlyNumbers(long int* d_num, long int* d_den, long int* d_c_vet, int size, int inc)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + inc;
	int j;

	if (i < size) {
		for (j = i + 1; j < size; j++) {
			if ((d_num[i] == d_num[j]) && (d_den[i] == d_den[j]))
				d_c_vet[i]++;
		}
	}
}

void friendly_numbers(long int start, long int end) {
	int devid;
	unsigned long int inc = 0;
	struct cudaDeviceProp prop;
	long int *d_the_num, *d_num, *d_den, *d_c_vet;

	cudaSetDevice(0);

	cudaGetDevice(&devid);
	cudaGetDeviceProperties(&prop, devid);

	int c=0;
	long int last = end - start + 1;
	size_t size = last * sizeof(long int);
	int nBlocks = 4*((prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount)/prop.maxThreadsPerBlock);
	int threads = prop.maxThreadsPerBlock;
	int qtdGrid = last / (nBlocks * threads)+1;

	long int *the_num;
	long int *num;
	long int *den;
	long int *c_vet;

	the_num = (long int*) malloc(size);
	num = (long int*) malloc(size);
	den = (long int*) malloc(size);
	c_vet = (long int*) malloc(size);

	cudaMalloc((void**)&d_the_num, size);						
	cudaMalloc((void**)&d_num, size);																																		
	cudaMalloc((void**)&d_den, size);
	cudaMalloc((void**)&d_c_vet, size);

	long int i;

	for (i = 0; i < last; i++) {
		c_vet[i] = 0;
	}

	cudaMemcpy(d_the_num, the_num, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_num, num, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_den, den, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c_vet, c_vet, size, cudaMemcpyHostToDevice);

	for (i = 0; i < qtdGrid; i++) {
		NumDemSearch<<<nBlocks, threads>>>(d_the_num, d_num, d_den, start, end, last, inc);
		inc += nBlocks * threads;
	}
	
	inc = 0;

	for (i = 0; i < qtdGrid; i++) {
		SumFriendlyNumbers<<<nBlocks, threads>>>(d_num, d_den, d_c_vet, last, inc);
		inc += nBlocks * threads;
	}

	cudaMemcpy(c_vet, d_c_vet, size, cudaMemcpyDeviceToHost);

	for (i = 0; i < last; i++) {
		c += c_vet[i];
	}

	printf("Founded %d pairs of mutually friendly numbers\n", c);

	free(the_num);
	free(num);
	free(den);
	free(c_vet);

	cudaFree(d_the_num);
	cudaFree(d_num);
	cudaFree(d_den);
	cudaFree(d_c_vet);
}

int main(int argc, char **argv) {
	long int start;
	long int end;

	if (argc != 3)
	{
		printf("Wrong number of arguments, needed 2 \n");
		return EXIT_SUCCESS;
	}

	start   = atoi(argv[1]);
	end     = atoi(argv[2]);

	printf("Number %ld to %ld\n", start, end);
	friendly_numbers(start, end);

	return EXIT_SUCCESS;
}
