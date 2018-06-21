/*
This code implements KNN algorithm: singlethread version,  using CUDA Framework and using omp
*/
#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include <device_functions.h>
#include <ctime>
#include <chrono>
#include <string>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <clocale>
#include <list>
#include <set>
#include <cctype>




// ---------------------------------
// BEGIN OF USER AREA

// Debug level, possible values: 0 - 5, 5 is highest
// Highest level will cause EXTREMELY detailed output (the whole array will be printed)
__constant__ const int DEBUG_LEVEL = 4;


//// Array size for initialization, used only in inputArray functiont
//__constant__ const int G_ARRAY_SIZE = 8192;
//
//// Number of threads inside of block
__constant__ const int BLOCK_SIZEX = 4;
__constant__ const int BLOCK_SIZEY = 256;

__constant__ const int GRID_SIZEX = 64;
__constant__ const int GRID_SIZEY = 4; // must be 1 for unknown reason. maybe atomicAdd problem

__constant__ const int BIG_VALUE = 1000000000; // must be 1 for unknown reason. maybe atomicAdd problem
const int WARMUP = 1;
const int REPEAT_TIME = 4;



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

template <typename T>
void outputArrayOnly(T * _arr, int arr_size) {
	if (DEBUG_LEVEL >= 5) {
		std::wcout << "Rank array: ";
		for (int i = 0; i < arr_size; i++) {
			std::wcout << _arr[i] << ", ";
		}
		std::wcout << std::endl;
	}



}


template <typename T>
void outputMatrix(T* source, int len1, int len2) {
	if (DEBUG_LEVEL >= 5) {
		for (int i = 0; i < len1; i++) {
			std::cout << i << ") ";
			for (int j = 0; j < len2; j++)
			{
				std::cout << source[i*len2 + j] << ", ";

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

__global__ void distanceKernel(const double *inputMatrixTest, const double *inputMatrixTrain, double *distanceMatrix, int testSize, int trainSize, int attrSize)
{
	int baseStrideWidthX = blockDim.x * gridDim.x;
	int baseStrideWidthY = blockDim.y * gridDim.y;
	int threadTileIndX = blockIdx.x * blockDim.x + threadIdx.x;
	int threadTileIndY = blockIdx.y * blockDim.y + threadIdx.y;


	double tmpVal = 0;
	double sum = 0;


	int targetIndexX = 0;
	int targetIndexY = 0;



	size_t xStride = 0;
	size_t yStride = 0;
	do {
		yStride = 0;

		do {
			targetIndexX = threadTileIndX + xStride;
			targetIndexY = threadTileIndY + yStride;

			// calculate  arr[x] - arr[y]
			for (size_t i = 0; i < attrSize - 1; i++)
			{
				tmpVal = inputMatrixTest[targetIndexX*attrSize + i] - inputMatrixTrain[targetIndexY*attrSize + i];
				sum += tmpVal * tmpVal;

			}

			distanceMatrix[targetIndexX*trainSize + targetIndexY] = sum;
			__syncthreads();
			//perThreadSum += tmpVal;
			tmpVal = 0;
			sum = 0;

			yStride += baseStrideWidthY;

		} while (yStride < trainSize);

		__syncthreads();


		xStride += baseStrideWidthX;
	} while (xStride < testSize);

}


__global__ void reductionStep(double *distanceMatrix, int *outputIndexes, double *distanceMins, int *distanceMinInds, int testSize, int trainSize)
{
	int baseStrideWidthX = blockDim.x * gridDim.x;// 16
	int baseStrideWidthY = blockDim.y * gridDim.y;  // 16
	int threadTileIndX = blockIdx.x * blockDim.x + threadIdx.x; // 1
	int threadTileIndY = blockIdx.y * blockDim.y + threadIdx.y; // 0


	__shared__ int sdata[BLOCK_SIZEY][BLOCK_SIZEX]; //"static" shared memory  16*16

	__shared__ int sind[BLOCK_SIZEY][BLOCK_SIZEX]; //"static" shared memory 16*16
	unsigned int tidx = threadIdx.x;    //1
	unsigned int tidy = threadIdx.y;    //0

	if (threadTileIndX < testSize && threadTileIndY < trainSize)
	{
		int targetIndexX = 0;
		int targetIndexY = 0;

		double prevMin = BIG_VALUE;
		double prevMinInd = 0;

		size_t xStride = 0;
		size_t yStride = 0;
		do {
			xStride = 0;

			do {
				targetIndexX = threadTileIndX + xStride;   //17
				targetIndexY = yStride + threadTileIndY;  //0

														  // calculate  arr[x] - arr[y]
				sdata[tidy][tidx] = distanceMatrix[targetIndexY*trainSize + targetIndexX];
				sind[tidy][tidx] = targetIndexY * trainSize + targetIndexX;

				__syncthreads();
				for (unsigned int s = blockDim.x / 2; s >= 1; s = s / 2)
				{
					if (tidx < s)
					{
						if (sdata[tidy][tidx] > sdata[tidy][tidx + s])
						{
							sdata[tidy][tidx] = sdata[tidy][tidx + s];
							sind[tidy][tidx] = sind[tidy][tidx + s];
						}

					}
					__syncthreads();
				}


				if (tidx == 0) {
					double x = sdata[tidy][0];
					int ind = sind[tidy][0];
					if (x < prevMin) {
						prevMin = x;
						prevMinInd = ind;

					}
				}

				xStride += baseStrideWidthX;
				__syncthreads();
			} while (xStride < trainSize);

			__syncthreads();

			if (tidx == 0)
			{
				distanceMins[targetIndexY* gridDim.x + blockIdx.x] = prevMin;
				distanceMinInds[targetIndexY* gridDim.x + blockIdx.x] = prevMinInd;
			}
			__syncthreads();
			if (threadTileIndX == 0) {
				double min = BIG_VALUE;
				int minInd = 0;
				double value;
				for (size_t i = 0; i < gridDim.x; i++)
				{
					value = distanceMins[targetIndexY* gridDim.x + i];

					if (value < min) {
						min = value;
						minInd = distanceMinInds[targetIndexY* gridDim.x + i];
					}
				}
				outputIndexes[targetIndexY] = minInd - targetIndexY * trainSize;  // save only column part of index
				distanceMatrix[minInd] = BIG_VALUE;
			}
			__syncthreads();
			prevMin = BIG_VALUE;
			yStride += baseStrideWidthY;
		} while (yStride < testSize);
	}
}

__global__ void voteKernel(const double *inputMatrixTrain, int* indexes, int* votes, int testSize, int attrSize, int classCount)
{
	int baseStrideWidthX = blockDim.x * gridDim.x;
	int baseStrideWidthY = blockDim.y * gridDim.y;
	int threadTileIndX = blockIdx.x * blockDim.x + threadIdx.x;
	int threadTileIndY = blockIdx.y * blockDim.y + threadIdx.y;

	int targetIndexX = 0;
	int targetIndexY = 0;

	// !To avoid 1 atomic add !
	/*
	__shared__ int sums[BLOCK_SIZEX][BLOCK_SIZEY];
	*/
	size_t xStride = 0;
	size_t yStride = 0;

	do {
		xStride = 0;
		do {

			targetIndexX = threadTileIndX + xStride;
			targetIndexY = threadTileIndY + yStride;
			int classValue = (int)inputMatrixTrain[indexes[targetIndexX] * attrSize + attrSize - 1];
			if (targetIndexY == classValue) {

				//votes[(targetIndexX * classCount) + classValue] = classValue;
				atomicAdd(&(votes[(targetIndexX * classCount) + classValue]), 1);
			}

			xStride += baseStrideWidthX;
		} while (xStride < testSize);
		yStride += baseStrideWidthY;
	} while (yStride < classCount);

}


//***************************************************************************
// singlethread version
//***************************************************************************
template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}
bool is_number(const std::string& s)
{
	return !s.empty() && std::find_if(s.begin(),
		s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}
std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}
std::set<std::string> listOfClasses = {};
class DataEntry {

private:

public:
	std::vector<double> attributes;
	int EntryClass;

	DataEntry(std::vector<std::string> input) {
		//attributes = new double[input.size() - 1];
		for (int i = 0; i < input.size() - 1; i++) {
			if (is_number(input[i].c_str())) {

				attributes.push_back(atof(input[i].c_str()));
			}
			else 
			{
				int code = std::hash<std::string>()(input[i].c_str());
				attributes.push_back(code);
			}
			//attributes[i] =(atof(input[i].c_str()));
		}
		std::string aclass = input[input.size() - 1].c_str();
		listOfClasses.insert(aclass);

		std::set<std::string>::iterator it = listOfClasses.find(aclass);
		//EntryClass = atoi(input[input.size() - 1].c_str());
		EntryClass= std::distance(listOfClasses.begin(), it);
	}
	DataEntry() {
		EntryClass = 0;

	}
};


size_t TrainingDataSize;
size_t TestDataSize;
DataEntry* TrainingData;	
DataEntry* TestData;
std::string datainfo;

/*
Reads data from  ","-delimited file.  First rowNumber/2 rows will be in training set and rest of lines will be in test set
*/
void ReadData(char* path, int rowNumber) {

	std::vector<std::string> row;

	std::vector<int> counters = std::vector<int>(50);


	std::string s = path;
	// var rows = File.ReadLines(path, System.Text.Encoding.Default);
	std::vector<std::string> rows;
	std::ifstream file(s, std::ios::in);
	while (!(file.eof()) && rowNumber>0)
	{
		getline(file, s);
		rows.push_back(s);
		rowNumber--;
	};
	file.close();
	if (rowNumber > 0) {
		std::cout << "Количество строк, указанное вами больше, чем размер файла. Считан весь файл", "Предупреждение\n\r";
	}
	TrainingDataSize = rows.size() / 2;
	TrainingData = new DataEntry[rows.size() / 2];
	for (int i = 0; i<rows.size() / 2; i++) {
		row = split(rows[i], ',');
		TrainingData[i] = *(new DataEntry(row));
		counters[TrainingData[i].EntryClass]++;

	}
	datainfo = datainfo + "Training: \n\r";
	for (size_t i = 0; i < counters.size(); i++)
	{
		if (counters.at(i) != 0) {
			datainfo = datainfo + "class " + std::to_string(i) + " - " + std::to_string(counters.at(i)) + "\n\r";

		}

	}
	counters.clear();
    counters = std::vector<int>(50);

	TestDataSize = rows.size() - rows.size() / 2;
	TestData = new DataEntry[rows.size() - rows.size() / 2];
	int j = 0;
	for (int i = rows.size() / 2; i < rows.size(); i++)
	{
		row = split(rows[i], ',');

		TestData[j] = *(new DataEntry(row));

		counters[TestData[j].EntryClass]++;
		j++;
	}
	datainfo = datainfo + "Test: \n\r";
	for (size_t i = 0; i < counters.size(); i++)
	{
		if (counters.at(i) != 0) {
			datainfo = datainfo + "class " + std::to_string(i) + " - " + std::to_string(counters.at(i)) + "\n\r";

		}

	}
	std::cout << datainfo;

}


void PrepareForGPU(DataEntry* dataEntries, double* targetMatrix, int datasetSize, int dataSetAttr) {
	int attrSize = dataSetAttr - 1;
	for (size_t i = 0; i < datasetSize; i++)
	{
		for (size_t j = 0; j < attrSize; j++)
		{

			targetMatrix[i*dataSetAttr + j] = dataEntries[i].attributes[j];
		}
		targetMatrix[i*dataSetAttr + attrSize] = dataEntries[i].EntryClass;


	}
}

void NormalizeTraining() {
	int sz = TrainingData[0].attributes.size();
	double* mins = new double[sz];
	double* maxs = new double[sz];

	for (size_t i = 0; i <sz; i++) {
		double min = TrainingData[0].attributes[i];
		double max = TrainingData[0].attributes[i];

		for (size_t j = 1; j < TrainingDataSize; j++) {
			if (TrainingData[j].attributes[i]<min) {
				min = TrainingData[j].attributes[i];
			}
			if (TrainingData[j].attributes[i]>max) {
				max = TrainingData[j].attributes[i];
			}
		}

		mins[i] = min;
		maxs[i] = max;
	}
	for (int j = 0; j<TrainingDataSize; j++)
	{
		for (int i = 0; i < TrainingData[j].attributes.size(); i++)
		{
			TrainingData[j].attributes[i] = (TrainingData[j].attributes[i] - mins[i]) / (maxs[i] - mins[i]);
		}
	}

}


void NormalizeTest() {
	int sz = TestData[0].attributes.size();
	double* mins = new double[sz];
	double* maxs = new double[sz];

	for (size_t i = 0; i <sz; i++) {
		double min = TestData[0].attributes[i];
		double max = TestData[0].attributes[i];

		for (size_t j = 1; j < TestDataSize; j++) {
			if (TestData[j].attributes[i]<min) {
				min = TestData[j].attributes[i];
			}
			if (TestData[j].attributes[i]>max) {
				max = TestData[j].attributes[i];
			}
		}

		mins[i] = min;
		maxs[i] = max;
	}
	for (int j = 0; j<TestDataSize; j++)
	{
		for (int i = 0; i < TestData[j].attributes.size(); i++)
		{
			TestData[j].attributes[i] = (TestData[j].attributes[i] - mins[i]) / (maxs[i] - mins[i]);
		}
	}

}


double Distance(DataEntry a, DataEntry b) {
	double res = 0;
	/*int id = omp_get_thread_num();
	int numt = omp_get_num_threads();
	printf("Thread(%d) of(%d) threads alive\n", id, numt);*/
	for (int i = 0; i < a.attributes.size(); i++) {
		res = res + std::pow((a.attributes[i] - b.attributes[i]), 2);
	}

	res = std::sqrt(res);
	return res;

}


double vote_fun(DataEntry* neighbours, int nbCount, int DataClass, DataEntry x) {
	double res = 0;
	for (size_t i = 0; i < nbCount; i++) {
		if (neighbours[i].EntryClass == DataClass) {
			res = res + 1 / std::pow(Distance(x, neighbours[i]), 2);
		}

	}
	return res;
}


int KNN(int k, DataEntry o) {
	DataEntry* neighbours = new DataEntry[k];
	std::vector<double> distances;
	double maxDist = 0;
	double Dist = 0;

	for (size_t j = 0; j < TrainingDataSize; j++) {
		Dist = Distance(TrainingData[j], o);
		distances.push_back(Dist);
		if (Dist > maxDist) maxDist = Dist;
	}
	for (size_t i = 0; i < k; i++) {
		double min = distances[0];
		size_t min_ind = 0;
		for (size_t j = 1; j < distances.size(); j++) {
			if (distances[j] < min) {
				min = distances[j];
				min_ind = j;
			}
		}
		neighbours[i] = TrainingData[min_ind];
		distances[min_ind] = maxDist;
	}

	int MaxClass = 0;
	double MaxVoteRes = 0;
	double tmp;
	std::vector<int> Classes = { 2, 4 }; // Задано явно

	for (int i = 0; i < Classes.size(); i++) {
		tmp = vote_fun(neighbours, k, Classes[i], o);
		if (tmp > MaxVoteRes) {
			MaxClass = Classes[i];
			MaxVoteRes = tmp;

		}
	}

	return MaxClass;
}


int KNNParallel(int k, DataEntry o) {
	DataEntry* neighbours = new DataEntry[k];

	std::vector<double> distances;

	double maxDist = 0;
	double Dist = 0;
	int j = 0;
	int cnt = 0;
#pragma omp parallel for shared (maxDist) private (j,Dist)
	for (j = 0; j < TrainingDataSize; j++) {
		Dist = Distance(TrainingData[j], o);
#pragma omp critical
		distances.push_back(Dist);

#pragma omp critical
		if (Dist > maxDist) maxDist = Dist;
	}

	for (size_t i = 0; i < k; i++) {
		double min = distances[0];
		size_t min_ind = 0;
		for (size_t j = 1; j < distances.size(); j++) {
			if (distances[j] < min) {
				min = distances[j];
				min_ind = j;
			}
		}
		neighbours[i] = TrainingData[min_ind];

		distances[min_ind] = maxDist;
	}


	int MaxClass = 0;
	double MaxVoteRes = 0;
	double tmp;
	std::vector<int> Classes = { 2, 4 }; // ßâíî çàäàíî!!

	for (int i = 0; i < Classes.size(); i++) {
		tmp = vote_fun(neighbours, k, Classes[i], o);
		if (tmp > MaxVoteRes) {
			MaxClass = Classes[i];
			MaxVoteRes = tmp;

		}
	}

	return MaxClass;
}

int main()
{

	cudaError_t cudaStatus = cudaSuccess;
	int rowNum = 2048;
	int testSize = rowNum / 2;
	int trainSize = rowNum / 2;




	ReadData("D:\\C#\\data mining\\input.with_dates.csv", rowNum);
	NormalizeTraining();
	NormalizeTest();
	int CLASS_COUNT = listOfClasses.size() ;
	int attrSize = TrainingData[0].attributes.size()+1;
	int k = 4;


	int TP = 0;
	int TN = 0;
	int FP = 0;
	int FN = 0;
	int TPP = 0;
	int TNP = 0;
	int FPP = 0;
	int FNP = 0;


	double *inputArrTest;
	double *d_inputArrTest;
	double *inputArrTrain;
	double *d_inputArrTrain;
	double *outputDistances;
	double *d_outputDistances;
	double *d_blockDistanceMin;
	int *d_blockDistanceMinInd;
	int *d_outputVotes;
	int *outputVotes;


	int  *d_outputIndexes;
	int *outputIndexes;

	int inputTestSize = testSize * attrSize * sizeof(double);
	int inputTrainSize = trainSize * attrSize * sizeof(double);
	int distancesSize = trainSize * testSize * sizeof(double);
	int outputIndexesSize = testSize * sizeof(int);
	int blockDistanceMinSize = GRID_SIZEX * testSize * sizeof(double);
	int blockDistanceMinIndSize = GRID_SIZEX * testSize * sizeof(int);

	int outputVotesSize = CLASS_COUNT * testSize * sizeof(int);

	if (DEBUG_LEVEL >= 1)
	{
		printInfo(cudaStatus);
	}



	inputArrTest = (double *)malloc(inputTestSize);
	inputArrTrain = (double *)malloc(inputTrainSize);
	outputDistances = (double *)malloc(distancesSize);
	outputIndexes = (int *)malloc(outputIndexesSize);

	outputVotes = (int *)malloc(outputVotesSize);

	PrepareForGPU(TestData, inputArrTest, testSize, attrSize);
	PrepareForGPU(TrainingData, inputArrTrain, trainSize, attrSize);


	std::cout << "Test: " << std::endl;
	outputMatrix(inputArrTest, testSize, attrSize);
	std::cout << "Train: " << std::endl;
	outputMatrix(inputArrTrain, trainSize, attrSize);

	// allocate GPU space for arrays
	cudaMalloc((void**)&d_inputArrTest, inputTestSize);
	cudaMalloc((void**)&d_inputArrTrain, inputTrainSize);
	cudaMalloc((void**)&d_outputDistances, distancesSize);
	cudaMalloc((void**)&d_outputIndexes, outputIndexesSize);
	cudaMalloc((void**)&d_blockDistanceMin, blockDistanceMinSize);
	cudaMalloc((void**)&d_blockDistanceMinInd, blockDistanceMinIndSize);
	cudaMalloc((void**)&d_outputVotes, outputVotesSize);





	/***********************************************************************************
	*              GPU VERSION
	************************************************************************************/
	std::cout << "******************************************************************" << "\n";
	std::cout << "GPU Version:" << "\n";
	std::cout << "******************************************************************" << "\n";




	long totalTime = 0;
	for (size_t r = 0; r < REPEAT_TIME + WARMUP; r++)
	{


#pragma region time
		auto start = std::chrono::high_resolution_clock::now();

#pragma endregion
		// copy input array to device
		cudaStatus = cudaMemcpy(d_inputArrTest, inputArrTest, inputTestSize, cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(d_inputArrTrain, inputArrTrain, inputTrainSize, cudaMemcpyHostToDevice);




		dim3 grid(GRID_SIZEX, GRID_SIZEY);
		dim3 blocks(BLOCK_SIZEX, BLOCK_SIZEY);
		distanceKernel << <grid, blocks >> >(d_inputArrTest, d_inputArrTrain, d_outputDistances, testSize, trainSize, attrSize);



		for (size_t i = 0; i < k; i++)
		{
			reductionStep << <grid, blocks >> >(d_outputDistances, d_outputIndexes, d_blockDistanceMin, d_blockDistanceMinInd, testSize, trainSize);

			voteKernel << <grid, blocks >> >(d_inputArrTrain, d_outputIndexes, d_outputVotes, testSize, attrSize, CLASS_COUNT);
		}



		//cudaStatus = cudaMemcpy( outputDebug, d_blockDistanceMin, blockDistanceMinSize, cudaMemcpyDeviceToHost);

		if (cudaErrorOccured(cudaStatus)) system("pause");


		cudaStatus = cudaMemcpy(outputVotes, d_outputVotes, outputVotesSize, cudaMemcpyDeviceToHost);



#pragma region time
		auto finish = std::chrono::high_resolution_clock::now();
		if (r > WARMUP) {
			totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
		}

#pragma endregion

	}
	std::cout << "averange time :" << totalTime / REPEAT_TIME << "ms\n";
#pragma region checkCorrectness
	TP = 0;
	TN = 0;
	FP = 0;
	FN = 0;
	TPP = 0;
	TNP = 0;
	FPP = 0;
	FNP = 0;
	cudaStatus = cudaMemcpy(inputArrTest, d_inputArrTest, inputTestSize, cudaMemcpyDeviceToHost);
	int classTwo = 2;
	int classFour = 4;
	for (int i = 0; i < testSize; i++)
	{
		int a = inputArrTest[i* attrSize + attrSize - 1];
		// int b = KNN(4, TestData[i]);
		int max = 0;
		int maxInd = 0;

		for (size_t j = 0; j < CLASS_COUNT; j++)
		{
			if (max < outputVotes[i*CLASS_COUNT + j]) {
				max = outputVotes[i*CLASS_COUNT + j];
				maxInd = j;
			}
		}

		int b = maxInd;
		if (a == b) TP++;
		else {
			FP++;
		}
		///printf("%d) Настоящий класс: %d | Предсказанный %d \n\r", i, a, b);
	}
	printf("TP - %d | FP - %d \n\r", TP, FP);

#pragma endregion



	cudaStatus = cudaMemcpy(outputIndexes, d_outputIndexes, outputIndexesSize, cudaMemcpyDeviceToHost);
	if (cudaErrorOccured(cudaStatus)) system("pause");

	cudaStatus = cudaMemcpy(outputDistances, d_outputDistances, distancesSize, cudaMemcpyDeviceToHost);
	std::cout << "distances:" << "\n";
	outputMatrix(outputDistances, testSize, trainSize);


	std::cout << "votes:" << "\n";

	outputMatrix(outputVotes, testSize, CLASS_COUNT);

	cudaFree(d_outputDistances);
	cudaFree(d_inputArrTest);
	cudaFree(d_inputArrTrain);
	cudaFree(d_blockDistanceMin);
	cudaFree(d_blockDistanceMinInd);
	cudaFree(d_outputIndexes);
	cudaFree(d_outputVotes);
	int* cheackArr = new int[TestDataSize];
/*
	std::cout << "******************************************************************" << "\n";
	std::cout << "Omp Version:" << "\n";
	std::cout << "******************************************************************" << "\n";

	int id = omp_get_thread_num();
	int numt = omp_get_num_threads();
	printf("Thread(%d) of(%d) threads alive\n", id, numt);
	int* cheackArr = new int[TestDataSize];

	//#pragma parallel for
	totalTime = 0;
	for (size_t r = 0; r < REPEAT_TIME + WARMUP; r++)
	{
#pragma region time

		auto start = std::chrono::high_resolution_clock::now();

#pragma endregion
		TP = 0;
		TN = 0;
		FP = 0;
		FN = 0;
		TPP = 0;
		TNP = 0;
		FPP = 0;
		FNP = 0;
		for (int i = 0; i < TestDataSize; i++)
		{
			int a = TestData[i].EntryClass;
			// int b = KNN(4, TestData[i]);

			int b = KNNParallel(k, TestData[i]);
			cheackArr[i] = b;

			///printf("%d) Настоящий класс: %d | Предсказанный %d \n\r", i, a, b);
		}



#pragma region time
		auto finish = std::chrono::high_resolution_clock::now();
		if (r > WARMUP) {
			totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
		}

#pragma endregion

	}
	std::cout << "averange time :" << totalTime / REPEAT_TIME << "ms\n";

#pragma region checkSingleVersion
	for (size_t i = 0; i < TestDataSize; i++)
	{
		int b = cheackArr[i];
		int a = TestData[i].EntryClass;
		if (a == b) { 
			TP++;
	    }
	else {
		FP++;
		}

	}
	printf("TP - %d | FP - %d \n\r", TP, FP);


#pragma endregion

	*/
	std::cout << "******************************************************************" << "\n";
	std::cout << "Single Thread Version:" << "\n";
	std::cout << "******************************************************************" << "\n";

	totalTime = 0;
	for (size_t r = 0; r < REPEAT_TIME + WARMUP; r++)
	{
#pragma region time

		auto start = std::chrono::high_resolution_clock::now();

#pragma endregion
		TP = 0;
		TN = 0;
		FP = 0;
		FN = 0;
		TPP = 0;
		TNP = 0;
		FPP = 0;
		FNP = 0;
		for (int i = 0; i < TestDataSize; i++)
		{
			int a = TestData[i].EntryClass;
			cheackArr[i] = KNN(k, TestData[i]);
			//int b = KNNParallel(4, TestData[i]);

			///printf("%d) Настоящий класс: %d | Предсказанный %d \n\r", i, a, b);
		}


#pragma region time
		auto finish = std::chrono::high_resolution_clock::now();
		if (r > WARMUP) {
			totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
		}

#pragma endregion

	}

	std::cout << "averange time :" << totalTime / REPEAT_TIME << "ms\n";
#pragma region checkSingleVersion
	for (size_t i = 0; i < TestDataSize; i++)
	{
		int b = cheackArr[i];
		int a = TestData[i].EntryClass;
		if (a == b) {
			TP++;
		}
		else {
			FP++;
		}

	}
	printf("TP - %d | FP - %d \n\r", TP, FP);


#pragma endregion



	system("pause");


	return 0;
}

