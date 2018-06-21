/*
Практическая работа №3. Алгоритмы редукции с помощью AMP C++
Реализовать 4 варианта алгоритма редукции и сравнить их эффективность по времени вычислений. 
В качестве задачи редукции можно использовать: сложение элементов вектора, умножение элементов вектора, 
поиск максимального или минимального элемента вектора.

1. Редукция без блочной декомпозиции
В этой реализации не используется блочно-статическая (разделяемая) память и не выделяются блоки потоков.
В коде host-программы выполняется цикл с уменьшением шага смещения (stride).
На каждой итерации цикла выполняется запуск GPU-вычислений.
В ядерной функции потоки выполняют редукцию пар элементов, находящихся на расстоянии stride. 
Результат помещается в позицию первого элемента:

a[i] = a[i] + a[i + stride];

2.  Редукция без блочной декомпозиции с окном

В этой реализации каждый поток, выполняющий редукцию на данной итерации,
осуществляет операцию над несколькими элементами (число элементов определяет ширину окна).
Например,

a[i] = a[i] + a[i + stride] + a[i + 2* stride] + a[i + 3*stride];

3.  Блочный алгоритм с расхождением

В этой реализации выполняется блочная декомпозиция. 
Каждый блок выполняет редукцию части массива, используя разделяемую память.
Внутри ядерной функции выполняется цикл с последовательным увеличением сдвига для редукции пар элементов. 
На каждой итерации этого цикла число работающих потоков сокращается вдвое, а сдвиг увеличивается вдвое.
В конечном итоге единственный поток (с номером 0) осуществляет редукцию пар элементов
и записывает результат в нулевую позицию в разделяемой памяти. 
Затем поток сохраняет результат в глобальной памяти в соответствии с абсолютным положением первого потока в общем массиве.

После завершения GPU-вычислений число элементов сокращается в M раз, где M – число элементов в блоке.
При этом остается столько элементов, сколько блоков образуется с учетом общего числа элементов (N / M).
Это число может быть еще очень большим, чтобы поместиться в одном блоке.
Поэтому необходимо выполнить несколько итераций в host-программе с запуском GPU-вычислений с уменьшением числа
обрабатываемых элементов и сокращением числа блоков. 
Цикл можно завершить, когда число элементов станет меньше размера блока. Конечную редукцию можно выполнить уже на CPU.

4. Блочный каскадный алгоритм

В этой реализации используется «окно обработки» в ядерной функции.
Если в предыдущей версии, каждый поток блока загружал единственный элемент из глобальной памяти в разделяемую память,
то в этой реализации каждый поток сохраняет в разделяемой памяти результат редукции нескольких элементов (например, восьми).

Кроме этого, в этой реализации потоки выполняют редукцию элементов разделяемой памяти, таким образом,
чтобы на первой итерации были заняты все первые M/2-потоков; на второй итерации заняты все первые M/4-потоков. 
Для этого шаг смещения (stride) последовательно уменьшается, начиная с M/2 и заканчивая 1.

В этой реализации за счет окна редукции можно ограничиться единственным запуском GPU-вычислений.
*/
#include "stdafx.h"
#include "stdafx.h"
#include <stdio.h>
#include <amp.h>
#include <amp_math.h>
#include <iostream>
#include "timer.h"
#include <omp.h>
#include <ppl.h>
#include "GpuConcLab3.h"


using namespace concurrency;
using namespace std;


void printInfo(vector<accelerator> source) {
	for (size_t i = 0; i < source.size(); i++)
	{
		auto a = source[i];
		wcout << "Description: " << a.description << endl;
		wcout << "Path: " << a.device_path << endl;
		wcout << "Memory: " << a.dedicated_memory << endl;
		wcout << "Is emulated: " << a.is_emulated << endl;
		wcout << endl;
	}
}
template <typename T>
void printVector(T* source, int len) {
	for (int i = 0; i < len; i++) {
		cout << source[i] << ", ";

	}
	cout << endl;
}
template <typename T>
void printVectorSum(T* source, int len) {
	T result = 0;
	for (int i = 0; i < len; i++) {
		result += source[i];
	}
	printf("                                   control value : %f \n", result);
//	cout << "                                   control value :" << result << endl;
}
template <typename T>
void printMatrixSum(T** source, int len) {
	T result = 0;
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++)
		{

			result += source[i][j];
		}


	}
	cout << "control value :" << result << endl;
}
template <typename T>
void printMatrix(T* source, int len1, int len2) {
	T result = 0;
	for (int i = 0; i < len1; i++) {
		for (int j = 0; j < len2; j++)
		{
			cout << source[i*len1 + j] << ", ";

		}
		cout << "|" << endl;
	}

}
template <typename T>
void ResetVector(T* matrix, int sz) {

	for (int i = 0; i < sz; i++) {
			matrix[i] = i+1;
	}
}
template <typename T>
void Zeros(T* matrix, int sz) {
	for (int i = 0; i < sz; i++) {
			matrix[i] = 0;
	}
}
template <typename T>
void printVectorSumOnly(T* source, int len) {
	T result = 0;
	for (int i = 0; i < len; i++) {
		result += source[i];
	}
	//cout  << result << endl;
	printf("%f \n", result);

}

template<typename T>
T vectorSum(T* source, int len) {
	T result = 0;
	for (int i = 0; i < len; i++) {
		result += source[i];
	}
	return result;
}
template<typename T>
T vectorSumWindow(T* source,int ind, int end) {
	T result = 0;
	for (int i = ind; i < end; i++) {
		result += source[i];
	}
	return result;
}


/*
1. Редукция без блочной декомпозиции
В этой реализации не используется блочно-статическая (разделяемая) память и не выделяются блоки потоков.
В коде host-программы выполняется цикл с уменьшением шага смещения (stride).
На каждой итерации цикла выполняется запуск GPU-вычислений.
В ядерной функции потоки выполняют редукцию пар элементов, находящихся на расстоянии stride.
Результат помещается в позицию первого элемента:

a[i] = a[i] + a[i + stride];


*/
void VectorSumReduction1(double array[], double out[], const int n)
{
	const int arraySize = n;
	unsigned elementCount = n;
	double tail_elem = (elementCount % 2) ? array[elementCount - 1] : 0;
	concurrency::array<double, 1> arr(arraySize, array);
	array_view<double, 1> av_tail_sum(1, &tail_elem);

	for (unsigned s = elementCount / 2; s > 0; s /= 2)
	{
		parallel_for_each(concurrency::extent<1>(s),
			[=, &arr](index<1> idx) restrict(amp)
		{
			arr[idx] = arr[idx] + arr[idx + s];

			if ((idx[0] == s - 1) && (s & 0x1) && (s != 1))
			{
			av_tail_sum[0] += arr[s - 1];
			}
		});		
	}
	std::vector<double> result(1);
	concurrency::copy(arr.section(0, 1), out);
	av_tail_sum.synchronize();
	out[0] += tail_elem;

}

/*

2.  Редукция без блочной декомпозиции с окном

В этой реализации каждый поток, выполняющий редукцию на данной итерации,
осуществляет операцию над несколькими элементами (число элементов определяет ширину окна).
Например,

a[i] = a[i] + a[i + stride] + a[i + 2* stride] + a[i + 3*stride];

*/
void VectorSumReduction2(double array[], double out[], const int n, const int window_width)
{
	const int arraySize = n;

	int elementCount = n;
	int prev_s = elementCount;
	double tail_sum =0;
	if ((elementCount % window_width) != 0 && elementCount > window_width)
	{
		for (int i = 1; i <= elementCount % window_width; i++)
		{
			tail_sum += array[elementCount-i];
		}
		
	}

	array_view<double, 1> av_tail_sum(1, &tail_sum);
	concurrency::array<double, 1> arr(arraySize, array);

	for (int s = elementCount / window_width; s > 0; s /= window_width)
	{
		parallel_for_each(concurrency::extent<1>(s),
			[=, &arr](index<1> idx) restrict(amp)
		{
			double sum = 0.f;
			for (int i = 0; i < window_width; i++)
			{
				sum += arr[idx + i * s];
			}
			arr[idx] = sum;

			if ((idx[0] == s - 1) && ((s % window_width) != 0) && (s > window_width))
			{

				for (int i = 1; i <= s % window_width; i++)
				{
					av_tail_sum[0] += arr[s-i];
				}
			}
		});
		prev_s = s;

	}
	concurrency::copy(arr.section(0, prev_s), out);
	av_tail_sum.synchronize();
	out[0] += tail_sum;

}


/*
3.  Блочный алгоритм с расхождением

В этой реализации выполняется блочная декомпозиция.
Каждый блок выполняет редукцию части массива, используя разделяемую память.
Внутри ядерной функции выполняется цикл с последовательным увеличением сдвига для редукции пар элементов.
На каждой итерации этого цикла число работающих потоков сокращается вдвое, а сдвиг увеличивается вдвое.
В конечном итоге единственный поток (с номером 0) осуществляет редукцию пар элементов
и записывает результат в нулевую позицию в разделяемой памяти.
Затем поток сохраняет результат в глобальной памяти в соответствии с абсолютным положением первого потока в общем массиве.

После завершения GPU-вычислений число элементов сокращается в M раз, где M – число элементов в блоке.
При этом остается столько элементов, сколько блоков образуется с учетом общего числа элементов (N / M).
Это число может быть еще очень большим, чтобы поместиться в одном блоке.
Поэтому необходимо выполнить несколько итераций в host-программе с запуском GPU-вычислений с уменьшением числа
обрабатываемых элементов и сокращением числа блоков.
Цикл можно завершить, когда число элементов станет меньше размера блока. Конечную редукцию можно выполнить уже на CPU.
*/

inline bool check_tiled_precondition(unsigned tile_size, unsigned element_count)
{
	while ((element_count % tile_size) == 0)
	{
		element_count /= tile_size;
	}
	return element_count < tile_size;
}
template <int tileSize>
void VectorSumReduction3(double array[], double out[], const int n)
{

	const int arraySize = n;
	unsigned elementCount = n;
	
	if (!check_tiled_precondition(tileSize, elementCount))
	{
		std::cout << "Warning, reduction_tiled_1 is not designed for the current problem size." << std::endl;
	}
	
	concurrency::array<double, 1> arr_1(elementCount, array);
	
	concurrency::array<double, 1> arr_2((elementCount / tileSize) != 0 ?
		                                (elementCount / tileSize) :
		                                 1);

	
	concurrency::array_view<double, 1> av_src(arr_1);
	concurrency::array_view<double, 1> av_dst(arr_2);
	av_dst.discard_data();


	// Reduce using parallel_for_each as long as the sequence length
	// is evenly divisable to the number of threads in the tile.
	while ((elementCount % tileSize) == 0)
	{
		parallel_for_each(concurrency::extent<1>(elementCount).tile<tileSize>(),
			[=](tiled_index<tileSize> tidx) restrict(amp)
		{
			tile_static double tile_data[tileSize];

			unsigned local_idx = tidx.local[0];
			tile_data[local_idx] = av_src[tidx.global];
			tidx.barrier.wait();


			for (unsigned s = 1; s < tileSize; s *= 2)
			{
				if (local_idx % (2 * s) == 0)
				{
					tile_data[local_idx] += tile_data[local_idx + s];
				}

				tidx.barrier.wait();
			}
			// Store the tile result in the global memory.
			if (local_idx == 0)
			{
				av_dst[tidx.tile] = tile_data[0];
			}


		});
		elementCount = elementCount / tileSize;
		std::swap(av_src, av_dst);
		av_dst.discard_data();

	}
	double* result = new double[elementCount];
	concurrency::copy(av_src.section(0, elementCount), result);
	out[0] = vectorSum(result, elementCount);
	


}
/*
4. Блочный каскадный алгоритм

В этой реализации используется «окно обработки» в ядерной функции.
Если в предыдущей версии, каждый поток блока загружал единственный элемент из глобальной памяти в разделяемую память,
то в этой реализации каждый поток сохраняет в разделяемой памяти результат редукции нескольких элементов(например, восьми).

Кроме этого, в этой реализации потоки выполняют редукцию элементов разделяемой памяти, таким образом,
чтобы на первой итерации были заняты все первые M / 2 - потоков; на второй итерации заняты все первые M / 4 - потоков.
Для этого шаг смещения(stride) последовательно уменьшается, начиная с M / 2 и заканчивая 1.

В этой реализации за счет окна редукции можно ограничиться единственным запуском GPU - вычислений.
*/
// tileSize - размер блока , tileCount - число работающих блоков,  batchSize - сколько поток обработает блоков.
template <int tileSize, int tileCount, int batchSize>
double VectorSumReduction4(double array[], const int n)
{
	
	unsigned elementCount = n;
	unsigned stride = tileSize * tileCount * batchSize;
	
	double tailSum = 0.0;


	unsigned tailLength = elementCount % stride;
	if (tailLength != 0) {
		tailSum = vectorSumWindow(array, n - tailLength, n);
		elementCount -= tailLength;
		if (elementCount == 0) {
			return tailSum;
		}
	}

	concurrency::array<double, 1> arr(elementCount, array);
	concurrency::array<double, 1> partial_result(tileCount);
	parallel_for_each(concurrency::extent<1>(tileCount * tileSize).tile<tileSize>(),
		[=, &arr, &partial_result](tiled_index<tileSize> tidx) restrict(amp)
	{
		tile_static double tile_data[tileSize];

		unsigned local_idx = tidx.local[0];

		unsigned targetStartIndex = (tidx.tile[0] * batchSize * tileSize) + local_idx;
		tile_data[local_idx] = 0.0;
	
			double temp = 0.0;
		do
		{		
			for (unsigned i = 0; i < batchSize; i++)
			{
				temp += arr[targetStartIndex + tileSize * i];
			}

			targetStartIndex += stride;
		} while (targetStartIndex < elementCount);
		tile_data[local_idx] = temp;
		tidx.barrier.wait();
		
		// Reduce local result in tileData to tileData[0]
		for (int localStride = tileSize/2; localStride > 0; localStride /= 2)
		{
			if (local_idx < localStride) {
				tile_data[local_idx] += tile_data[local_idx + localStride];
			}
	
			tidx.barrier.wait();
		}
		

		// Store result to partial result in global memory 
		if (local_idx == 0) {
			partial_result[tidx.tile[0]] = tile_data[0];
		}

	});

	// collect result from worker tiles on the CPU

	 double* result = new double[tileCount];
	//std::vector<float> v_partial_result(tileCount);
	//concurrency::copy(partial_result, v_partial_result.begin());
	// return  vectorSum(result, elementCount) + tailSum;
	 concurrency::copy(partial_result, result);
	
	return  vectorSum(result, tileCount) + tailSum;


}


void multiplyMatrixByMatrixAmp(double matrix1[], double matrix2[], double out[], const int l, const int m, const int n) {

	array_view<const double, 2> a(l, m, matrix1);
	array_view<const double, 2> b(m, n, matrix1);
	array_view<double, 2> c(l, n, out);

	c.discard_data();
	parallel_for_each(
		c.extent,
		[=](index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];

		double temp = 0;
		for (int i = 0; i < m; i++) {
			temp += a(row, i)*b(i, col);;
		}
		c[idx] = temp;
	});
	c.synchronize();



}

void multiplicationMatrixWithBlocks(double matrix1[], double matrix2[], double out[], const int m)
{
	static const int tileSize = 32;
	const int matrixSize = m;


	concurrency::array<double, 2> mat1(matrixSize, matrixSize, matrix1);
	concurrency::array<double, 2> mat2(matrixSize, matrixSize, matrix2);
	concurrency::array<double, 2> outData(matrixSize, matrixSize);
	parallel_for_each(outData.extent.tile<tileSize, tileSize>(),
		[=, &mat1, &mat2, &outData](tiled_index<tileSize, tileSize> tidx) restrict(amp)
	{
		int row = tidx.local[0];
		int col = tidx.local[1];

		double sum = 0.0;
		tile_static double sA[tileSize][tileSize];
		tile_static double sB[tileSize][tileSize];
		for (int i = 0; i < matrixSize; i += tileSize)
		{
			sA[row][col] = mat1(tidx.global[0], col + i);
			sB[row][col] = mat2(row + i, tidx.global[1]);

			tidx.barrier.wait();

			for (int k = 0; k < tileSize; k++)
			{
				sum += sA[row][k] * sB[k][col];
			}
			tidx.barrier.wait();
		}
		outData[tidx.global] = sum;
	});


	try {
		concurrency::copy(outData, out);
		//outData.associated_accelerator_view.wait();
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

}


void multiplicationMatrixWithBlocksWithoutShared(double matrix1[], double matrix2[], double out[], const int m)
{
	static const int tileSize = 32;
	const int matrixSize = m;


	concurrency::array<double, 2> mat1(matrixSize, matrixSize, matrix1);
	concurrency::array<double, 2> mat2(matrixSize, matrixSize, matrix2);
	concurrency::array<double, 2> outData(matrixSize, matrixSize);
	parallel_for_each(outData.extent.tile<tileSize, tileSize>(),
		[=, &mat1, &mat2, &outData](tiled_index<tileSize, tileSize> tidx) restrict(amp)
	{
		int row = tidx.global[0];
		int col = tidx.global[1];
		double temp = 0.0;

		for (int k = 0; k < matrixSize; k++) {
			temp += mat1[row][k] * mat2[k][col];
		}
		//tidx.barrier.wait();
		outData[tidx.global] = temp;

	});

	concurrency::copy(outData, out);


}




void AddArraysAmp(int n, const float* pA, const float* pB, float* pC)
{
	array_view<const float, 1> a(n, pA);
	array_view<const float, 1> b(n, pB);
	array_view<float, 1> sum(n, pC);
	sum.discard_data();

	parallel_for_each(sum.extent,
		[=](index<1> idx) restrict(amp)
	{
		sum[idx] = a[idx] + b[idx];
	}
	);
	sum.synchronize();
}


void AddArraysSingle(int n, const float* pA, const float* pB, float* pC)
{
	for (int i = 0; i < n; i++)
	{
		pC[i] = pA[i] + pB[i];
	}

}

int main()
{
	Timer timer = Timer();
	auto a = accelerator::default_accelerator;
	//**************************************************************************************************************************************************
	//          print info
	//**************************************************************************************************************************************************
	vector<accelerator> all = accelerator::get_all();
	printInfo(all);

	int N = 32768;
	double* arr = new double[N];
	double* out= new double[1];


	std::cout << "*************************************************************************" << endl;
	std::cout << "                       Reduction 1: Two elements for each thread " << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region Reduction1
	try {
		std::cout << "Stride: ";
		ResetVector(arr, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		//printVector(arr, N);
		timer.start();
		VectorSumReduction1(arr, out, N);

		timer.stop();
		std::printf("%lld \n", timer.elapsed_microseconds());
		// sum of first row
		std::printf(" Result : %f \n", out[0]);
		printVectorSum(arr, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion

	std::cout << "*************************************************************************" << endl;
	std::cout << "                        Reduction 2: Sliding window " << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region Reduction2
	try {
		std::cout << "Windowed: ";
		int window_width = 100;
		out = new double[N];
		Zeros(out, N);
		ResetVector(arr, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		VectorSumReduction2(arr, out, N,window_width);

		timer.stop();
		std::printf("%lld \n", timer.elapsed_microseconds());
		// sum of first row
		std::printf(" Result : "); 
		printVectorSumOnly(out, N);
		printVectorSum(arr, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion

	std::cout << "*************************************************************************" << endl;
	std::cout << "                        Reduction 3: Tiled-1 " << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region Reduction2
	try {
		std::cout << "Tiled-1: ";
		out = new double[N];
		int tileSize = 1024;
		Zeros(out, N);
		ResetVector(arr, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();


		VectorSumReduction3<32>(arr, out, N);

		timer.stop();
		std::printf("%lld \n", timer.elapsed_microseconds());
		// sum of first row
		std::printf(" Result : %f", out[0]);

		//printVectorSumOnly(out, N);
		printVectorSum(arr, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion

	std::cout << "*************************************************************************" << endl;
	std::cout << "                        Reduction 4: Tiled-with fixed tile workers " << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region Reduction2
	try {
		std::cout << "Tiled-2: ";
		out = new double[N];
		int tileSize = 32;
		//Zeros(out, N);
		ResetVector(arr, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();


		double sumRes = VectorSumReduction4<32,4,8>(arr, N);

		timer.stop();
		std::printf("%lld \n", timer.elapsed_microseconds());
		// sum of first row
		std::printf(" Result : %f", sumRes);

		//printVectorSumOnly(out, N);
		printVectorSum(arr, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion



	system("pause");


    return 0;
}

