#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <SFML/Graphics.hpp>
#include <GPUMandelbrot.h>

dim3 BlockSize(16, 16);

//ADD HLSL SHADER LATER -> More performant as no copies back and forth between cpu and gpu

__global__ 
void MandelbrotKernelOne(int* OutputTexture, const int2 TextureSize, const double2 Offset, const double2 DrawArea, const int MaxIterations)
{   
    unsigned int PixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int PixelY = blockIdx.y * blockDim.y + threadIdx.y;

    double MandelbrotCoordX = Offset.x + DrawArea.x * (((double)PixelX / (double)TextureSize.x) - 0.5);
    double MandelbrotCoordY = Offset.y + DrawArea.y * (((double)PixelY / (double)TextureSize.y) - 0.5);

    double x = 0;
    double y = 0;
    double x2 = 0;
    double y2 = 0;

    int Iteration = 0;

    while (x2 + y2 < 4. && Iteration < MaxIterations)
    {
        y = 2 * x * y + MandelbrotCoordY;
        x = x2 - y2 + MandelbrotCoordX;
        x2 = x * x;
        y2 = y * y;
        Iteration++;
    };

    if (Iteration == MaxIterations) Iteration = 64;

    OutputTexture[PixelY * TextureSize.x + PixelX] =
        ((unsigned int)((0.5f * sin(0.1f * Iteration) + 0.5f) * 255)) +
        ((unsigned int)((0.5f * sin(0.1f * Iteration + 2.094f) + 0.5f) * 255) << 8) +
        ((unsigned int)((0.5f * sin(0.1f * Iteration + 4.188f) + 0.5f) * 255) << 16) +
        ((unsigned int)255 << 24);
    
    /*
    OutputTexture[PixelY * TextureSize.x + PixelX] =
        ((unsigned int)(Iteration) +
        ((unsigned int)(Iteration) << 8) +
        ((unsigned int)(Iteration) << 16) +
        ((unsigned int)255 << 24));
    */
}

__host__
GPUColorTexture* Setup(unsigned int Width, unsigned int Height)
{
    cudaSetDevice(0);

    int CorrectedWidth = (int)std::ceilf((float)Width / (float)BlockSize.x) * BlockSize.x;
    int CorrectedHeight = (int)std::ceilf((float)Height / (float)BlockSize.y) * BlockSize.y;

    GPUColorTexture* Texture = new GPUColorTexture(CorrectedWidth, CorrectedHeight);

    return Texture;
}

__host__
void RunGPUMandelbrot(GPUColorTexture* TargetTexture, const sf::Vector2d Offset, const sf::Vector2d DrawArea, const int MaxIterations)
{       
    dim3 NumBlocks(TargetTexture->Width / BlockSize.x, TargetTexture->Height / BlockSize.y);
    int2 Size { TargetTexture->Width, TargetTexture->Height };
    double2 GPUOffset { Offset.x, Offset.y };
    double2 GPUDrawArea{ DrawArea.x, DrawArea.y };

    MandelbrotKernelOne<<<NumBlocks, BlockSize>>>(TargetTexture->GPUOutputTexture, Size, GPUOffset, GPUDrawArea, MaxIterations);
    
    auto cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    cudaStatus = cudaMemcpy(TargetTexture->CPUTexture, TargetTexture->GPUOutputTexture, TargetTexture->Width * TargetTexture->Height * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}
