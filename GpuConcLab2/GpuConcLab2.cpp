/*
Практическая работа №1.
Реализуйте следующие задачи с помощью технологии AMP C++
1) Получение информации об устройствах
2) Сложение двух векторов
3) Умножение матрицы на число
4) Транспонирование матрицы
5) Матричное умножение.
*/

#include "stdafx.h"
#include <stdio.h>
#include <ppl.h>
#include <amp.h>
#include <amp_math.h>
#include <iostream>
#include "timer.h"
#include <omp.h>
#include <ppl.h>
#include "GpuConcLab2.h"


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
	cout << "                                   control value :" << result << endl;
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
		cout <<"|"<< endl;
	}

}
template <typename T>
void ResetMatrix(T* matrix, int sz) {

	for (int i = 0; i < sz; i++) {
		for (int j = 0; j < sz; j++)
		{
			matrix[sz*i + j] = i;
		}
	}
}
template <typename T>
void Zeros(T* matrix, int sz) {

	for (int i = 0; i < sz; i++) {
		for (int j = 0; j < sz; j++)
		{
			matrix[sz*i + j] = 0;
		}
	}
}


//Matrix transponse
void transponseMatrixSingle(double matrix[], const int m)
{
	for (int rows = 0; rows < m; rows++) {
		for (int cols = rows + 1; cols < m; cols++) {
			swap(matrix[cols*m + rows], matrix[rows*m + cols]);
		}
	}
}
void transponseMatrixAmp(double matrix[], const int m)
{
	array_view<double, 2> a(m, m, matrix);
	parallel_for_each(
		a.extent,
		[=](index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];
		double temp = 0;
		if (row < col) {
			temp = a(row, col);
			a(row, col) = a(col, row);
			a(col, row) = temp;
		}
	});
	a.synchronize();
}






void transponseMatrixWithBlocks(double matrix[], const int m)
{
	static const int tileSize = 32;
	const int matrixSize = m;


	concurrency::array<double, 2> inData(matrixSize, matrixSize, matrix);
	concurrency::array<double, 2> outData(matrixSize, matrixSize);

	parallel_for_each(inData.extent.tile<tileSize, tileSize>(),
		[=, &inData, &outData](tiled_index<tileSize, tileSize> tidx) restrict(amp)
	{
		tile_static double localData[tileSize][tileSize];
		localData[tidx.local[1]][tidx.local[0]] = inData[tidx.global];
		tidx.barrier.wait();
		index<2> outIdx(
			index<2>(tidx.tile_origin[1], tidx.tile_origin[0]) 
			       + tidx.local);
		outData[outIdx] = localData[tidx.local[0]][tidx.local[1]];
	});
	try {
		//vector<int> v(96);
    
		concurrency::copy(outData, matrix);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

}



void transponseMatrixWithBlocksWithoutShared(double matrix[], const int m)
{
	static const int tileSize = 32;
	const int matrixSize = m;


	concurrency::array<double, 2> inData(matrixSize, matrixSize, matrix);
	concurrency::array<double, 2> outData(matrixSize, matrixSize);

	parallel_for_each(inData.extent.tile<tileSize, tileSize>(),
		[=, &inData, &outData](tiled_index<tileSize, tileSize> tidx) restrict(amp)
	{

		outData[tidx.global] = inData[tidx.global[1]][tidx.global[0]];
		tidx.barrier.wait();
	});
	try {
		//vector<int> v(96);
		concurrency::copy(outData, matrix);
		//outData.associated_accelerator_view.wait();
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

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
	concurrency::array<double, 2> mat2(matrixSize, matrixSize,matrix2);
	concurrency::array<double, 2> outData(matrixSize, matrixSize);
	parallel_for_each(outData.extent.tile<tileSize, tileSize>(),
		[=, &mat1, &mat2, &outData](tiled_index<tileSize, tileSize> tidx) restrict(amp)
	{
			int row = tidx.local[0];
			int col = tidx.local[1];

			double sum = 0.0;
			tile_static double sA[tileSize][tileSize];
			tile_static double sB[tileSize][tileSize];
			for (int i = 0; i < matrixSize; i+= tileSize)
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
			outData[tidx.global]=sum;	
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

void multiplicationMatrixEnlarged(double matrix1[], double matrix2[], double out[], const int m)
{
	static const int tileSize = 1024;
	const int matrixSize = m;


	concurrency::array<double, 2> mat1(matrixSize, matrixSize, matrix1);
	concurrency::array<double, 2> mat2(matrixSize, matrixSize, matrix2);
	concurrency::array<double, 1> idx_arr(matrixSize);
	concurrency::array<double, 2> outData(matrixSize, matrixSize);
	parallel_for_each(idx_arr.extent,
		[=, &mat1, &mat2, &outData](index<1> idx) restrict(amp)
	{
		int row = idx[0];
		double temp = 0.0;
		for (int i = 0; i < matrixSize; i++) {
			temp = 0.0;
			for (int k = 0; k < matrixSize; k++) {
				temp += mat1[row][k] * mat2[k][i];
			}
			outData[row][i] = temp;
		}
		//tidx.barrier.wait();
		//outData[tidx.global] = temp;

	});

	concurrency::copy(outData, out);

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

	int N = 640;
	double* matrix= new double[N*N];
	double* matrix2 = new double[N*N];
	double* matrix3 = new double[N*N];

	std::cout << "*************************************************************************" << endl;
	std::cout << "                             Transpose from first lab " << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region TransposeMatrix
	try {
		std::cout << "single: ";
		ResetMatrix(matrix, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		transponseMatrixSingle(matrix, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row

		printVectorSum(matrix, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

	try {
		std::cout << "amp: ";
		ResetMatrix(matrix, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		transponseMatrixAmp(matrix, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		printVectorSum(matrix, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}
#pragma endregion

	std::cout << "*************************************************************************" << endl;
	std::cout << "                             Transpose with blocks " << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region TransposeMatrixBlocks
	try {
		std::cout << "amp with blocks: ";
		ResetMatrix(matrix, N);
	
		//	printVector(vector1, N);
		//printVector(vector2, N);
	//	printMatrix(matrix, N, N);
		timer = Timer();
		timer.start();
		transponseMatrixWithBlocks(matrix, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		//printMatrix(matrix2, N, N);
		printVectorSum(matrix, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion

	std::cout << "*************************************************************************" << endl;
	std::cout << "                             Transpose with blocks without shared " << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region TransposeMatrixBlocksWithoutShared
	try {
		std::cout << "amp with blocks without shared: ";
		ResetMatrix(matrix, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		//printMatrix(matrix, N, N);
		timer = Timer();
		timer.start();
		transponseMatrixWithBlocksWithoutShared(matrix, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		//printMatrix(matrix2, N, N);
		printVectorSum(matrix, N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion

	std::cout << "*************************************************************************" << endl;
	std::cout << "                        Multiplication from First" << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region MultiFromFirst
	try {
		std::cout << "amp: ";
		ResetMatrix(matrix, N);
		ResetMatrix(matrix2, N);
		Zeros(matrix3, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		//printMatrix(matrix, N, N);
		timer = Timer();
		timer.start();
		multiplyMatrixByMatrixAmp(matrix, matrix2, matrix3, N, N, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		//printMatrix(matrix2, N, N);
		printVectorSum(matrix3, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion



	std::cout << "*************************************************************************" << endl;
	std::cout << "                        Multiplication with blocks" << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region MultiWithBlocks
	try {
		std::cout << "amp with blocks: ";
		ResetMatrix(matrix, N);
		ResetMatrix(matrix2, N);
		Zeros(matrix3, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		//printMatrix(matrix, N, N);
		timer = Timer();
		timer.start();
		multiplicationMatrixWithBlocks(matrix, matrix2, matrix3, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		//printMatrix(matrix2, N, N);
		printVectorSum(matrix3, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion

	std::cout << "*************************************************************************" << endl;
	std::cout << "                        Multiplication with blocks without shared" << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region MultiWithBlocks
	try {
		std::cout << "amp without shared: ";
		ResetMatrix(matrix, N);
		ResetMatrix(matrix2, N);
		Zeros(matrix3, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		//printMatrix(matrix, N, N);
		timer = Timer();
		timer.start();
		multiplicationMatrixWithBlocksWithoutShared(matrix, matrix2, matrix3, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		//printMatrix(matrix2, N, N);
		printVectorSum(matrix3, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion


	std::cout << "*************************************************************************" << endl;
	std::cout << "                        Multiplication enlarged" << endl;
	std::cout << "*************************************************************************" << endl;
#pragma region MultiWithBlocks
	try {
		std::cout << "enlarged: ";
		ResetMatrix(matrix, N);
		ResetMatrix(matrix2, N);
		Zeros(matrix3, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		//printMatrix(matrix, N, N);
		timer = Timer();
		timer.start();
		multiplicationMatrixEnlarged(matrix, matrix2, matrix3, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		//printMatrix(matrix2, N, N);
		printVectorSum(matrix3, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

#pragma endregion







	//auto a = [=]() restrict(amp) {
	//	int ans = fast_math::sinf(1.0f) == precise_math::sinf(1.0f);
	//};






	system("pause");

	//parallel_for(0, N, 
	//[=](int i) {
	//	a[i] = cosf(a[i]);
	//}

	//1. Выделить память на ускорителе
	//2. Скопировать данные в память ускорителя
	//3. Определить конфигурацию вычислений (решетку вычислений, блоки, размерность блоков, число блоков, распределение по блокам)
	//4. Запуск ядерной функции 
	//5. Ожидание завершения вычислений
	// https://msdn.microsoft.com/ru-ru/library/dd293608.aspx
	// см слайды с курса

	return 0;
}

