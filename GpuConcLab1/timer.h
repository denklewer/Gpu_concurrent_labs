#pragma once
#include <winnt.h>

class Timer {
private:
	LARGE_INTEGER StartingTime, EndingTime, Elapsed, Frequency;

public:
	Timer::Timer() {
		QueryPerformanceFrequency(&Frequency);
	}

	void Timer::start() {
		QueryPerformanceCounter(&StartingTime);
	}

	void Timer::stop() {
		QueryPerformanceCounter(&EndingTime);
	}

	LONGLONG Timer::elapsed_microseconds() {
		Elapsed.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
		Elapsed.QuadPart *= 1000000;
		Elapsed.QuadPart /= Frequency.QuadPart;
		return Elapsed.QuadPart;
	}

	double Timer::elapsed_seconds() {
		return (this->elapsed_microseconds() / 1000000.0);
	}

	Timer::~Timer() {}
};