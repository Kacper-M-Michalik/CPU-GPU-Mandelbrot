﻿#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <SFML/Graphics.hpp>
#include <GPUMandelbrot.h>

dim3 BlockSize(16, 16);

//Change to pass obj directly next time?
__global__ void MandelbrotKernelOne(int* OutputTexture, const int2 TextureSize, const double2 Offset, const double Zoom, const int MaxIterations)
{   
    unsigned int PixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int PixelY = blockIdx.y * blockDim.y + threadIdx.y;

    double ScaledX = Offset.x + Zoom * (((double)PixelX / (double)TextureSize.x) - 0.5);
    double ScaledY = Offset.y + Zoom * (((double)PixelY / (double)TextureSize.y) - 0.5);

    double x = 0;
    double y = 0;
    double x2 = 0;
    double y2 = 0;

    int Iteration = 0;

    while (x2 + y2 < 4. && Iteration < MaxIterations)
    {
        y = 2 * x * y + ScaledY;
        x = x2 - y2 + ScaledX;
        x2 = x * x;
        y2 = y * y;
        Iteration++;
    };

    OutputTexture[PixelY * TextureSize.x + PixelX] =
        ((unsigned int)((0.5f * sin(0.1f * Iteration) + 0.5f) * 255)) +
        ((unsigned int)((0.5f * sin(0.1f * Iteration + 2.094f) + 0.5f) * 255) << 8) +
        ((unsigned int)((0.5f * sin(0.1f * Iteration + 4.188f) * 255)) << 16) +
        ((unsigned int)255 << 24);
}

__global__ void MandelbrotKernelTwo(int* OutputTexture, const int2 TextureSize, const double2 Offset, const double Zoom, const int MaxIterations)
{
    unsigned int PixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int PixelY = blockIdx.y * blockDim.y + threadIdx.y;

    double ScaledX = Offset.x + Zoom * (((double)PixelX / (double)TextureSize.x) - 0.5);
    double ScaledY = Offset.y + Zoom * (((double)PixelY / (double)TextureSize.y) - 0.5);

    double x = 0;
    double y = 0;
    double x2 = 0;
    double y2 = 0;

    int Iteration = 0;

    while (Iteration < MaxIterations)
    {
        y = 2 * x * y + ScaledY;
        x = x2 - y2 + ScaledX;
        x2 = x * x;
        y2 = y * y;
        if (x2 + y2 < 4) Iteration++;
    };

    OutputTexture[PixelY * TextureSize.x + PixelX] =
        ((unsigned int)((0.5f * sin(0.1f * Iteration) + 0.5f) * 255)) +
        ((unsigned int)((0.5f * sin(0.1f * Iteration + 2.094f) + 0.5f) * 255) << 8) +
        ((unsigned int)((0.5f * sin(0.1f * Iteration + 4.188f) * 255)) << 16) +
        ((unsigned int)255 << 24);
}

__host__
GPUColorTexture* Setup(unsigned int Width, unsigned int Height)
{
    //(Width * Height) / 

    cudaSetDevice(0);

    int CorrectedWidth = (int)std::ceilf((float)Width / (float)BlockSize.x) * BlockSize.x;
    int CorrectedHeight = (int)std::ceilf((float)Height / (float)BlockSize.y) * BlockSize.y;

    GPUColorTexture* Texture = new GPUColorTexture(CorrectedWidth, CorrectedHeight);

    return Texture;

    /*
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    */
}

__host__
void RunGPUMandelbrot(GPUColorTexture* TargetTexture, const sf::Vector2d Offset, const double Zoom, const int MaxIterations)
{       
    dim3 NumBlocks(TargetTexture->Width / BlockSize.x, TargetTexture->Height / BlockSize.y);

    int2 Size { TargetTexture->Width, TargetTexture->Height };
    double2 GPUOffset { Offset.x, Offset.y };

    MandelbrotKernelOne<<<NumBlocks, BlockSize>>>(TargetTexture->GPUOutputTexture, Size, GPUOffset, Zoom, MaxIterations);

    auto cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    cudaStatus = cudaMemcpy(TargetTexture->CPUTexture, TargetTexture->GPUOutputTexture, TargetTexture->Width * TargetTexture->Height * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}