#pragma once
#include <SFML/Graphics.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

struct GPUColorTexture
{
    int Width;
    int Height;

    int* GPUOutputTexture;
    sf::Uint8* CPUTexture;

    GPUColorTexture(unsigned int SetWidth, unsigned int SetHeight)
    {
        Width = SetWidth;
        Height = SetHeight;
        cudaMalloc((void**)&GPUOutputTexture, Width * Height * sizeof(int));
        CPUTexture = (sf::Uint8*)malloc(sizeof(sf::Uint8) * Width * Height * 4);
    }

    ~GPUColorTexture()
    {
        cudaFree(GPUOutputTexture);
        free(CPUTexture);
    }
};

GPUColorTexture* Setup(unsigned int Width, unsigned int Height);

int plerp(double Base, double Max, double Point, double Power, int3 BaseColor, int3 MaxColor);

void RunMandelbrotKernel(GPUColorTexture* TargetTexture, const sf::Vector2d Offset, const sf::Vector2d DrawArea, const int MaxIterations);
void RunJuliaKernel(GPUColorTexture* TargetTexture, const sf::Vector2d Offset, const sf::Vector2d DrawArea, const sf::Vector2d C, const int MaxIterations);