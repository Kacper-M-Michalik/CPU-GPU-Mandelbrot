#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <SFML/Graphics.hpp>
#include <GPUMandelbrot.h>

dim3 BlockSize(16, 16);

//ADD HLSL SHADER LATER -> More performant as no copies back and forth between cpu and gpu

__global__ 
void MandelbrotKernel(int* OutputTexture, const int2 TextureSize, const double2 Offset, const double2 DrawArea, const int MaxIterations)
{   
    unsigned int PixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int PixelY = blockIdx.y * blockDim.y + threadIdx.y;

    double ComplexCoordX = Offset.x + DrawArea.x * (((double)PixelX / (double)TextureSize.x) - 0.5);
    double ComplexCoordY = Offset.y + DrawArea.y * (((double)PixelY / (double)TextureSize.y) - 0.5);

    double x = 0;
    double y = 0;
    double x2 = 0;
    double y2 = 0;

    int Iteration = 0;

    while (x2 + y2 < 4.0 && Iteration < MaxIterations)
    {
        y = 2 * x * y + ComplexCoordY;
        x = x2 - y2 + ComplexCoordX;
        x2 = x * x;
        y2 = y * y;
        Iteration++;
    }

    if (Iteration == MaxIterations && MaxIterations >= 64) Iteration = 64;

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


__global__
void JuliaKernel(int* OutputTexture, const int2 TextureSize, const double2 Offset, const double2 DrawArea, const double2 C, const int MaxIterations)
{
    unsigned int PixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int PixelY = blockIdx.y * blockDim.y + threadIdx.y;

    double x = Offset.x + DrawArea.x * (((double)PixelX / (double)TextureSize.x) - 0.5);;
    double y = Offset.y + DrawArea.y * (((double)PixelY / (double)TextureSize.y) - 0.5);;
    double x2 = x * x;
    double y2 = y * y;

    int Iteration = 0;

    while (x2 + y2 > 0.00001 && x2 + y2 < 4.0 && Iteration < MaxIterations)
    {
        y = 2 * x * y + C.y;
        x = x2 - y2 + C.x;
        x2 = x * x;
        y2 = y * y;
        Iteration++;
    }

    //Figure out some nice coloring algo, this is terrible.

    //convergent logic
    if (x2 + y2 < 0.00001)
    {
        //if (Iteration < 160) 
        //{
            //OutputTexture[PixelY * TextureSize.x + PixelX] = plerp(64.0, 160.0, (double)Iteration, 2, int3{ 0, 255, 0 }, int3{ 0, 241, 255 });
        //}
        //else 
        //{
            //OutputTexture[PixelY * TextureSize.x + PixelX] = plerp(160, (double)MaxIterations, (double)Iteration, 1.0, int3{ 0, 255, 0 }, int3{ 0, 241, 255 });
        //}
        //OutputTexture[PixelY * TextureSize.x + PixelX] = plerp(0, 1.0, 1.0, 1.0, int3{ 255, 0, 0 }, int3{ 255, 0, 0 });

        OutputTexture[PixelY * TextureSize.x + PixelX] = 4294967295u;
    }
    //iteration = MaxIter is not concrete proof of either
    else if (Iteration == MaxIterations) 
    {
        OutputTexture[PixelY * TextureSize.x + PixelX] = 0;
    }
    //Divergent
    else
    {
        if (Iteration <= 96) 
        {
            OutputTexture[PixelY * TextureSize.x + PixelX] = plerp(0.0, 96.0, (double)Iteration, 0.9, int3{ 4, 12, 44 }, int3{ 255, 255, 255 });
        }
        else
        {
            OutputTexture[PixelY * TextureSize.x + PixelX] = 4294967295u;
            //OutputTexture[PixelY * TextureSize.x + PixelX] = plerp(0.0, 1.0, 1.0, 1.0, int3{ 255, 255, 255 }, int3{ 255, 255, 255 });
        }
    }

   // OutputTexture[PixelY * TextureSize.x + PixelX] =
    //    ((unsigned int)((Iteration/64)*255)) +
    //    ((unsigned int)((Iteration/64)*255) << 8) +
    //    ((unsigned int)((Iteration/64)*255) << 16) +
    //    ((unsigned int)255 << 24);
           
    //OutputTexture[PixelY * TextureSize.x + PixelX] =
    //    ((unsigned int)(Iteration) +
    //    ((unsigned int)(Iteration) << 8) +
    //    ((unsigned int)(Iteration) << 16) + 
    //    ((unsigned int)255 << 24));
    
    
}

__device__
int plerp(double Base, double Max, double Point, double Power, int3 BaseColor, int3 MaxColor) 
{
    double t = fmin(pow((Point - Base) / (Max - Base), Power), 1.0);
    double OneMinus = 1 - t;

    return 
        ((unsigned int)(BaseColor.x * OneMinus + MaxColor.x * t) +
        ((unsigned int)(BaseColor.y * OneMinus + MaxColor.y * t) << 8) +
        ((unsigned int)(BaseColor.z * OneMinus + MaxColor.z * t) << 16) +
        (255u << 24));
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
void RunMandelbrotKernel(GPUColorTexture* TargetTexture, const sf::Vector2d Offset, const sf::Vector2d DrawArea, const int MaxIterations)
{       
    dim3 NumBlocks(TargetTexture->Width / BlockSize.x, TargetTexture->Height / BlockSize.y);
    int2 Size { TargetTexture->Width, TargetTexture->Height };
    double2 GPUOffset { Offset.x, Offset.y };
    double2 GPUDrawArea{ DrawArea.x, DrawArea.y };

    MandelbrotKernel<<<NumBlocks, BlockSize>>>(TargetTexture->GPUOutputTexture, Size, GPUOffset, GPUDrawArea, MaxIterations);
    
    auto cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    cudaStatus = cudaMemcpy(TargetTexture->CPUTexture, TargetTexture->GPUOutputTexture, TargetTexture->Width * TargetTexture->Height * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}

__host__
void RunJuliaKernel(GPUColorTexture* TargetTexture, const sf::Vector2d Offset, const sf::Vector2d DrawArea, const sf::Vector2d C, const int MaxIterations)
{
    dim3 NumBlocks(TargetTexture->Width / BlockSize.x, TargetTexture->Height / BlockSize.y);
    int2 Size{ TargetTexture->Width, TargetTexture->Height };
    double2 GPUOffset{ Offset.x, Offset.y };
    double2 GPUDrawArea{ DrawArea.x, DrawArea.y };
    double2 GPUC{ C.x, C.y };

    JuliaKernel<<<NumBlocks, BlockSize>>>(TargetTexture->GPUOutputTexture, Size, GPUOffset, GPUDrawArea, GPUC, MaxIterations);

    auto cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    cudaStatus = cudaMemcpy(TargetTexture->CPUTexture, TargetTexture->GPUOutputTexture, TargetTexture->Width * TargetTexture->Height * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
}
