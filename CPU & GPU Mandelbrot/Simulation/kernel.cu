#pragma once
#include <SFML/Graphics.hpp>
#include <GPUMandelbrot.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
CPU IMPLEMENTATION:
void OptimisedMandelbrot(ColorTexture* Texture, const Vector2f TileOffset, const Vector2f TileSize, const Vector2d Offset, const double Zoom, const int MaxIterations)
{
    float MandelbrotX;
    float MandelbrotY;

    int MaxPixelX = (int)TileOffset.x + (int)TileSize.x;
    int MaxPixelY = (int)TileOffset.y + (int)TileSize.y;

    float a = 0.1f;

    for (int PY = (int)TileOffset.y; PY < MaxPixelY; PY++)
    {
        for (int PX = (int)TileOffset.x; PX < MaxPixelX; PX++)
        {
            double ScaledX = Offset.x + ((double)PX / (double)Texture->Width) * Zoom;
            double ScaledY = Offset.y + ((double)PY / (double)Texture->Height) * Zoom;

            double x = 0;
            double y = 0;
            double x2 = 0;
            double y2 = 0;

            int Iteration = 0;

            while (x2 + y2 < 4.f && Iteration < MaxIterations)
            {
                y = 2 * x * y + ScaledY;
                x = x2 - y2 + ScaledX;
                x2 = x * x;
                y2 = y * y;
                Iteration++;
            };

            const int Index = 4 * (Texture->Width * PY + PX);
            Texture->Data[Index] = (Uint8)((0.5f * sin(a * Iteration) + 0.5f) * 255);
            Texture->Data[Index + 1] = (Uint8)((0.5f * sin(a * Iteration + 2.094f) + 0.5f) * 255);
            Texture->Data[Index + 2] = (Uint8)((0.5f * sin(a * Iteration + 4.188f) * 255));
            Texture->Data[Index + 3] = (Uint8)255;
        }
    }
}
*/

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void Setup()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

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

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
