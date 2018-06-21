/*
ENG: 
This is an implementations of rank sort using CUDA.

Реализовать один из алгоритмов сортировки с помощью CUDA:
2) Ранговая сортировка;
В реализации использовать разные подходы (атомарные операторы, работа с разделяемой памятью, работа с динамической памятью и др.)
*/
#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <device_functions.h>
#include <ctime>
#include <chrono>





// ---------------------------------
// BEGIN OF USER AREA

// Debug level, possible values: 0 - 5, 5 is highest
// Highest level will cause EXTREMELY detailed output (the whole array will be printed)
__constant__ const int DEBUG_LEVEL = 4;


//// Array size for initialization, used only in inputArray functiont
//__constant__ const int G_ARRAY_SIZE = 8192;
//
//// Number of threads inside of block
__constant__ const int BLOCK_SIZEX = 50;
__constant__ const int BLOCK_SIZEY = 50;

__constant__ const int GRID_SIZEX = 10000;
__constant__ const int GRID_SIZEY = 1; // must be 1 for unknown reason. maybe atomicAdd problem
const int WARMUP = 5;
const int REPEAT_TIME = 10;


void outputArray(int * _arr, int arr_size) {
	if (DEBUG_LEVEL >= 5) {
		std::wcout << "Array: ";
		for (int i = 0; i < arr_size; i++) {
			std::wcout << _arr[i] << ", ";
		}
		std::wcout << std::endl;
	}

	bool sorted = true;
	for (int i = 1; i < arr_size; i++) {
		if (_arr[i] < _arr[i - 1]) {
			sorted = false;
			break;
		}
	}

	if (DEBUG_LEVEL >= 1) std::wcout << "Array sorting check, sorted: " << std::boolalpha << sorted << std::endl;
}

void outputArrayOnly(int * _arr, int arr_size) {
	if (DEBUG_LEVEL >= 5) {
		std::wcout << "Rank array: ";
		for (int i = 0; i < arr_size; i++) {
			std::wcout << _arr[i] << ", ";
		}
		std::wcout << std::endl;
	}

	for (int i = 1; i < arr_size; i++) {
		if (_arr[i] < _arr[i - 1]) {
			break;
		}
	}

}


template <typename T>
void outputMatrix(T* source, int len1, int len2) {
	if (DEBUG_LEVEL >= 5) {
		for (int i = 0; i < len1; i++) {
			for (int j = 0; j < len2; j++)
			{
				std::cout << source[i*len1 + j] << ", ";

			}
			std::cout << "|" << std::endl;
		}
	}

}

// END OF USER AREA
// ---------------------------------


bool inline cudaErrorOccured(cudaError_t _cudaStatus) {
	if (_cudaStatus != cudaSuccess) {
		std::wcout << std::endl << std::endl
			<< "------------------------------"
			<< "CUDA error: " << _cudaStatus << std::endl;
		if (DEBUG_LEVEL >= 1) std::wcout << cudaGetErrorString(_cudaStatus) << std::endl;
		std::wcout
			<< "------------------------------"
			<< std::endl << std::endl;

		return true;
	}
	return false;
}


void random_ints(int* a, int N)
{
	int i;
	for (i = 0; i < N; ++i)
		//a[i] = rand();
		a[i] = (N - i) % 3 + 1;
	//a[i] = 5;

}

void printInfo(cudaError_t cudaStatus) {
	std::wcout << "CUDA realization of rank sorting algorithm" << std::endl;
	std::wcout << "Author: Denis Kleverov" << std::endl << std::endl;

	std::wcout << "CUDA information" << std::endl;
	int deviceCount = 0;
	cudaStatus = cudaGetDeviceCount(&deviceCount);
	if (cudaErrorOccured(cudaStatus)) {
		std::wcout << "!!!Error occured !!!" << std::endl << std::endl;

	}
	std::wcout << "Available CUDA device count: " << deviceCount << std::endl << std::endl;

	cudaDeviceProp devProps;
	for (int i = 0; i < deviceCount; i++) {
		cudaStatus = cudaGetDeviceProperties(&devProps, i);
		if (cudaErrorOccured(cudaStatus))
		{
			std::wcout << "!!!!Error occured !!!" << std::endl << std::endl;
		}


		std::wcout
			<< "Device #" << i << ", CUDA version: " << devProps.major << "." << devProps.minor
			<< ", integrated: " << std::boolalpha << devProps.integrated << std::endl
			<< "Name: " << devProps.name << std::endl
			<< "Clockrate: " << (double)devProps.clockRate / 1024 << "MHz" << std::endl
			<< "Total global memory: " << (double)devProps.totalGlobalMem / 1024 / 1024 / 1024 << "GB" << std::endl
			<< "Shared memory per block: " << (double)devProps.sharedMemPerBlock / 1024 << "KB" << std::endl
			<< "Warp size: " << devProps.warpSize << std::endl
			<< "Max threads per block: " << devProps.maxThreadsPerBlock << std::endl
			<< "Max threads dimension: ["
			<< devProps.maxThreadsDim[0] << ", "
			<< devProps.maxThreadsDim[1] << ", "
			<< devProps.maxThreadsDim[2] << "]" << std::endl
			<< "Max grid size: ["
			<< devProps.maxGridSize[0] << ", "
			<< devProps.maxGridSize[1] << ", "
			<< devProps.maxGridSize[0] << "]" << std::endl
			<< std::endl;
	}
	std::wcout << std::endl;
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void rankCalculateKernel(const int *inputArr, int *matrix, int *rankArr, int *outArr, int arrSize)
{
	int baseStrideWidthX = blockDim.x * gridDim.x;
	int baseStrideWidthY = blockDim.y * gridDim.y;
	int threadTileIndX = blockIdx.x * blockDim.x + threadIdx.x;
	int threadTileIndY = blockIdx.y * blockDim.y + threadIdx.y;


	int tmpVal;

	int targetIndexX = 0;
	int targetIndexY = 0;
	// !To avoid 1 atomic add !
	/*
	__shared__ int sums[BLOCK_SIZEX][BLOCK_SIZEY];
	*/

	__shared__ int sums[BLOCK_SIZEX];

	int perThreadSum = 0;

	if (threadTileIndX < arrSize && threadTileIndY < arrSize)
	{
		size_t xStride = 0;
		size_t yStride = 0;
		do {
			yStride = 0;
			do {
				targetIndexX = threadTileIndX + xStride;
				targetIndexY = yStride + threadTileIndY;

				// calculate  arr[x] - arr[y]
				tmpVal = inputArr[targetIndexX] - inputArr[targetIndexY];
				// add extra values for preventing duplicate values for rank  ( add triangle matrix of 1)
				if (targetIndexX > targetIndexY) {
					tmpVal += 1;
				}
				tmpVal = tmpVal > 0 ? 1 : 0;

				//write bool function to matrix;
				__syncthreads();
				matrix[(targetIndexX)*arrSize + targetIndexY] = tmpVal;
				__syncthreads();
				perThreadSum += tmpVal;
				tmpVal = 0;


				yStride += baseStrideWidthY;

			} while (yStride < arrSize);
			//accumulating perThreadSum into perRowSum  in 1 block

			__syncthreads();
			// !To avoid 1 atomic add !
			/*
			//sums[threadIdx.x][threadIdx.y] = perThreadSum;
			*/

			atomicAdd(&sums[threadIdx.x], perThreadSum);//  this atomic add can be avoided
			__syncthreads();


			// accumulate  row results from blocks into 1 value
			if (threadIdx.y == 0) {
				int tmp = 0;
				//!To avoid 1 atomic add !
				/*
				for (size_t i = 0; i < BLOCK_SIZEY; i++)
				{
				tmp += sums[threadIdx.x][i];
				}
				atomicAdd(&rankArr[targetIndexX], tmp);*/


				atomicAdd(&rankArr[targetIndexX], sums[threadIdx.x]);
				__syncthreads();

			}
			// accumulate  writting elements to the new places in output array according to rank
			if (threadIdx.y == 0) {
				int newIndex = rankArr[targetIndexX];
				outArr[newIndex] = inputArr[targetIndexX];
			}
			__syncthreads();
			//!To avoid 1 atomic add !
			/*
			sums[threadIdx.x][threadIdx.y] = 0;
			*/
			sums[threadIdx.x] = 0;
			__syncthreads();

			xStride += baseStrideWidthX;
			perThreadSum = 0;
		} while (xStride < arrSize);
	}
}






int main()
{

	cudaError_t cudaStatus = cudaSuccess;
	const int N = 10000;
	int *inputArr;
	int *rankArr;
	int *outputArr;
	int *d_inputArr;
	int *rankMatrix;
	int *d_rankMatrix;
	int *d_outputArr;
	int *d_rankArr;
	int arrSize = N * sizeof(int);
	int matrixSize = N * N * sizeof(int);

	if (DEBUG_LEVEL >= 1)
	{
		printInfo(cudaStatus);
	}


	// allocate GPU space for arrays
	cudaMalloc((void**)&d_inputArr, arrSize);
	cudaMalloc((void**)&d_rankMatrix, matrixSize);
	cudaMalloc((void**)&d_rankArr, arrSize);
	cudaMalloc((void**)&d_outputArr, arrSize);

	// allocate space for arrays in RAM
	printf("Allocating and initializing host arrays...\n\n");
	inputArr = (int *)malloc(arrSize); random_ints(inputArr, N);
	outputArray(inputArr, N);

	rankArr = (int *)malloc(arrSize);
	outputArr = (int *)malloc(arrSize);

	rankMatrix = (int *)malloc(matrixSize);
	/***********************************************************************************
	*              GPU VERSION
	************************************************************************************/
	printf("GPU version \n");
	long totalTime = 0;
	for (size_t i = 0; i < REPEAT_TIME + WARMUP; i++)
	{
		random_ints(inputArr, N);


#pragma region time
		auto start = std::chrono::high_resolution_clock::now();

#pragma endregion
		// copy input array to device
		cudaStatus = cudaMemcpy(d_inputArr, inputArr, N * sizeof(int), cudaMemcpyHostToDevice);
		dim3 grid(GRID_SIZEX, GRID_SIZEY);
		dim3 blocks(BLOCK_SIZEX, BLOCK_SIZEY);
		rankCalculateKernel << <grid, blocks >> > (d_inputArr, d_rankMatrix, d_rankArr, d_outputArr, N);




		cudaStatus = cudaMemcpy(outputArr, d_outputArr, N * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaErrorOccured(cudaStatus)) system("pause");
#pragma region time
		auto finish = std::chrono::high_resolution_clock::now();
		if (i > WARMUP) {
			totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
		}

#pragma endregion

	}
	std::cout << "averange time :" << totalTime / REPEAT_TIME << "ns\n";

	cudaStatus = cudaMemcpy(rankMatrix, d_rankMatrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaErrorOccured(cudaStatus)) system("pause");

	//outputMatrix(rankMatrix, N, N);

	cudaStatus = cudaMemcpy(rankArr, d_rankArr, N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaErrorOccured(cudaStatus)) system("pause");
	//	outputArrayOnly(rankArr, N);



	outputArray(outputArr, N);

	cudaFree(d_inputArr);
	cudaFree(d_rankArr);
	cudaFree(d_rankMatrix);
	cudaFree(d_outputArr);




	/***********************************************************************************
	*              CPU VERSION
	************************************************************************************/
	printf("Single thread version: \n");
	random_ints(inputArr, N);
	outputArray(inputArr, N);
	totalTime = 0;
	for (size_t i = 0; i < REPEAT_TIME + WARMUP; i++)
	{
		random_ints(inputArr, N);


#pragma region time
		auto start = std::chrono::high_resolution_clock::now();

#pragma endregion

		std::qsort(inputArr, N, sizeof(int), [](const void* a, const void* b)
		{
			int arg1 = *static_cast<const int*>(a);
			int arg2 = *static_cast<const int*>(b);
			if (arg1 < arg2) return -1;
			if (arg1 > arg2) return 1;
			return 0;
			//  return (arg1 > arg2) - (arg1 < arg2); // Возможное сокращение кода
			//  return arg1 - arg2; //Ошибочное сокращение (если присутствует INT_MIN)
		});
#pragma region time
		auto finish = std::chrono::high_resolution_clock::now();
		if (i > WARMUP) {
			totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
		}

#pragma endregion
	}
	outputArray(inputArr, N);
	std::cout << "averange time :" << totalTime / REPEAT_TIME << "ns\n";





	system("pause");
	return 0;
}

