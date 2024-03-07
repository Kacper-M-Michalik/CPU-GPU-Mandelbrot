#pragma once
#include <SFML/Graphics.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void Setup();

//void addKernel(int* c, const int* a, const int* b);

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);