#pragma once
#include <stdlib.h>
#include <string>

struct PerformanceCounter
{	
	std::string Name;
	double* TimeArray;
	int Size;
	int RealSize;
	int PlacementIndex;

	PerformanceCounter(int SetSize, std::string SetName) 
	{
		Name = SetName;
		PlacementIndex = 0;
		RealSize = 0;
		Size = SetSize;
		TimeArray = (double*)calloc(Size, Size * sizeof(double));

		for (int i = 0; i < Size; i++)
		{
			TimeArray[i] = 0;
		}
	}

	void AddTime(double Time) 
	{
		TimeArray[PlacementIndex] = Time;
		PlacementIndex++;
		if (PlacementIndex > RealSize) RealSize = PlacementIndex;
		if (PlacementIndex >= Size) PlacementIndex = 0;
	}

	double GetAverageTime() 
	{
		double Sum = 0;

		for (int i = 0; i < Size; i++)
		{
			Sum += TimeArray[i];
		}

		return Sum / (double)RealSize;
	}
};