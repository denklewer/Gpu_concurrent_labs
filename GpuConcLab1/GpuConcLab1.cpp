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
#include "GpuConcLab1.h"
#include <ppl.h>

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
void printVector(T* source, int len ) {
	for (int i = 0; i < len; i++) {
		cout << source[i] << ", " ;

	}
	cout << endl;
}
template <typename T>
void printVectorSum(T* source, int len) {
	T result = 0;
	for (int i = 0; i < len; i++) {
		result += source[i];

	}
	cout <<"                                   control value :" << result << endl;
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
		cout << endl;
	}

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
	for (size_t i = 0; i < n; i++)
	{
		pC[i] = pA[i] + pB[i];
	}

}

void AddArraysOmp(int n, const float* pA, const float* pB, float* pC)
{
    #pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		pC[i] = pA[i] + pB[i];
	}

}

void AddArraysPpl(int n, const float* pA, const float* pB, float* pC)
{

	parallel_for(0,n,1, [&](int i) {

		pC[i] = pA[i] + pB[i];
	});

}


void ResetMatrix(float* matrix, int sz) {

	for (int i = 0; i < sz; i++) {
		for (int j = 0; j < sz; j++)
		{
			matrix[sz*i + j] = i;
		}
	}
}

void Zeros(float* matrix, int sz) {

	for (int i = 0; i < sz; i++) {
		for (int j = 0; j < sz; j++)
		{
			matrix[sz*i + j] = 0;
		}
	}
}


 void multiplyMatrixSingle(float* matrix, const int m, float num)
 {
	 for (int i = 0; i < m; i++)
		 for (int j = 0; j < m; j++)
			 matrix[i*m+j] = matrix[i*m+j] * num;
	
}

 //Multiply matrix by number with AMP
  void multiplyMatrixAmp(float matrix[], const int m, float num)
 {
	 array_view<float, 2> a(m, m, matrix);
	 parallel_for_each(
		 a.extent,
		 [=](index<2> idx) restrict(amp) {
		 a[idx] *= num;
	 });
	 a.synchronize();
 }


  //Matrix transponse
  void transponseMatrixSingle(float matrix[], const int m)
  {
	  for (int rows = 0; rows < m; rows++) {
		  for (int cols = rows + 1; cols < m; cols++) {
			  swap(matrix[cols*m + rows], matrix[rows*m + cols]);
		  }
	  }
  }
  void transponseMatrixAmp(float matrix[], const int m)
  {
	  array_view<float, 2> a(m, m, matrix);
	  parallel_for_each(
		  a.extent,
		  [=](index<2> idx) restrict(amp) {
		  int row = idx[0];
		  int col = idx[1];
		  float temp = 0;
		  if (row < col) {
			  temp = a(row, col);
			  a(row, col) = a(col, row);
			  a(col, row) = temp;
		  }
	  });
	  a.synchronize();
  }

  void multiplyMatrixByMatrixSingle(float matrix1[], float matrix2[], float out[],const int l, const int m, const int n) {

	  for (int i = 0; i < l; i++)
	  {	
		  for (int j = 0; j < n; j++)
		  {
			  for (int k = 0; k < m; k++)
			  {
				  out[n * i + j] += matrix1[m * i + k] * matrix2[m * k + j];
			  }
		  }
	  }
  
  }

  void multiplyMatrixByMatrixAmp(float matrix1[], float matrix2[], float out[], const int l, const int m, const int n) {

	  array_view<const float, 2> a(l, m, matrix1);
	  array_view<const float, 2> b(m, n, matrix1);
	  array_view<float, 2> c(l, n, out);

	  c.discard_data();
	  parallel_for_each(
		  c.extent,
		  [=](index<2> idx) restrict(amp) {
		  int row = idx[0];
		  int col = idx[1];

		  float temp = 0;
		  for (int i = 0; i < m; i++) {
			  temp += a(row, i)*b(i, col);;			 
		  }
		  c[idx] = temp;
	  });
	  c.synchronize();



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

	  //***************************************************************************************************************************************************
	  //           Sum of vectors
	  //***************************************************************************************************************************************************
	  int N =	200;
#pragma region AddVectors
	  cout << "*************************************************************************" << endl;
	  cout << "                             Sum of vectors " << endl;
	  cout << "*************************************************************************" << endl;
	  float* vector1 = new float[N];
	  float* vector2 = new float[N];
	  float* vector3 = new float[N];

	  for (int i = 0; i < N; i++)
	  {
		  vector1[i] = i * (float)1;
		  vector2[i] = i * (float)1;
	  }


	  try {
		  cout << "single: ";

		  //	printVector(vector1, N);
		  //printVector(vector2, N);
		  timer = Timer();
		  timer.start();
		  AddArraysSingle(N, vector1, vector2, vector3);

		  timer.stop();
		  std::printf("%d \n", timer.elapsed_microseconds());
		  printVectorSum(vector3, N);
	  }
	  catch (exception* e) {
		  wcout << e->what() << endl;
	  }



	  try {
		  cout << "omp: ";

		  //	printVector(vector1, N);
		  //	printVector(vector2, N);
		  timer.start();
		  AddArraysOmp(N, vector1, vector2, vector3);
		  timer.stop();
		  std::printf("%d \n", timer.elapsed_microseconds());
		  printVectorSum(vector3, N);
		  //	printVector(vector3, N);
	  }
	  catch (exception* e) {
		  wcout << e->what() << endl;
	  }


	  try {
		  cout << "AMP: ";

		  //	printVector(vector1, N);
		  //	printVector(vector2, N);
		  timer.start();
		  AddArraysAmp(N, vector1, vector2, vector3);
		  timer.stop();
		  std::printf("%d \n", timer.elapsed_microseconds());
		  printVectorSum(vector3, N);
		  //	printVector(vector3, N);
	  }
	  catch (exception* e) {
		  wcout << e->what() << endl;
	  }
	  try {
		  cout << "Ppl : ";

		  //	printVector(vector1, N);
		  //	printVector(vector2, N);
		  timer.start();
		  AddArraysPpl(N, vector1, vector2, vector3);
		  timer.stop();
		  std::printf("%d \n", timer.elapsed_microseconds());
		  printVectorSum(vector3, N);
		  //	printVector(vector3, N);
	  }
	  catch (exception* e) {
		  wcout << e->what() << endl;
	  }
#pragma endregion

#pragma region MultiplyMatrixByNumber

	  cout << "*************************************************************************" << endl;
	  cout << "                             Multiply by number " << endl;
	  cout << "*************************************************************************" << endl;

	  float* matrix = new float[N*N];
	try {
		cout << "single: ";
		ResetMatrix(matrix, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		multiplyMatrixSingle(matrix, N, 7);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		printVectorSum(matrix, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

	try {
		cout << "amp: ";
		ResetMatrix(matrix, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
	    multiplyMatrixAmp(matrix, N, 7);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		printVectorSum(matrix, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}



#pragma endregion
	cout << "*************************************************************************" << endl;
	cout << "                             Transpose " << endl;
	cout << "*************************************************************************" << endl;
#pragma region TransposeMatrix

	try {
		cout << "single: ";
		ResetMatrix(matrix, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		transponseMatrixSingle(matrix, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		
		printVectorSum(matrix, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

	try {
		cout << "amp: ";
		ResetMatrix(matrix, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		transponseMatrixAmp(matrix, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		// sum of first row
		printVectorSum(matrix, N*N);
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}
#pragma endregion
	cout << "*************************************************************************" << endl;
	cout << "                             Matrix by Matrix " << endl;
	cout << "*************************************************************************" << endl;
#pragma region MultiplyMatrixByMatrix
	try {
		cout << "single: ";
		float* matrix1 = new float[N*N];
		float* matrix2 = new float[N*N];
		float* matrix3= new float[N*N];
		ResetMatrix(matrix1, N);
		ResetMatrix(matrix2, N);
		Zeros(matrix3, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		multiplyMatrixByMatrixSingle(matrix1,matrix2,matrix3, N, N, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
		printVectorSum(matrix3, N*N);


	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}

	try {
		cout << "amp: ";
		float* matrix1 = new float[N*N];
		float* matrix2 = new float[N*N];
		float* matrix3 = new float[N*N];
		ResetMatrix(matrix1, N);
		ResetMatrix(matrix2, N);
		Zeros(matrix3, N);
		//	printVector(vector1, N);
		//printVector(vector2, N);
		timer = Timer();
		timer.start();
		multiplyMatrixByMatrixAmp(matrix1, matrix2, matrix3, N, N, N);

		timer.stop();
		std::printf("%d \n", timer.elapsed_microseconds());
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

