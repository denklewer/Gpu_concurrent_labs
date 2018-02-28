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






int main()
{
	auto a = accelerator::default_accelerator;
	//???

	vector<accelerator> all = accelerator::get_all();
	Timer timer = Timer();
	int N = 10;
	float* vector = new float[N];
	

	//Желательно оборачивать GPU код в try..catch, потому что в ОС есть жесткий таймаут по ответу от ГПУ и если слишком долго выполняется, то exception
	try {
		timer.start();
		int i = 0;
		i = i++ + ++i;
		printf("i = %d \n", i);
		timer.stop();
		printf("elapsed = %d", timer.elapsed_microseconds());
	}
	catch (exception* e) {
		wcout << e->what() << endl;
	}


	//auto a = [=]() restrict(amp) {
	//	int ans = fast_math::sinf(1.0f) == precise_math::sinf(1.0f);
	//};




	float* arr = new float[N];
	for (int i = 0; i < N; i++)
	{
		arr[i] = i * (float)122.431;
	}

	for (int i = 0; i < N; i++)
	{
		arr[i] = cosf(arr[i]);
	}

	for (int i = 0; i < N; i++)
	{
		cout << arr[i] << '\n';
	}

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

