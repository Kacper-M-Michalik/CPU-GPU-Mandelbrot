#pragma once
#include <SFML/Graphics.hpp>
#include "mandelbrot.h"
#include <cmath>
#include <immintrin.h>

using namespace sf;

void SimpleMandelbrot(ColorTexture* Texture, const Vector2f TileOffset, const Vector2f TileSize, const Vector2d Offset, const double Zoom, const int MaxIterations)
{
	const int MaxPixelX = (int)TileOffset.x + (int)TileSize.x;
	const int MaxPixelY = (int)TileOffset.y + (int)TileSize.y;

	const float a = 0.1f;

	for (int PY = (int)TileOffset.y; PY < MaxPixelY; PY++)
	{
		for (int PX = (int)TileOffset.x; PX < MaxPixelX; PX++)
		{
			double ScaledX = Offset.x + Zoom * (((double)PX / (double)Texture->Width) - 0.5);
			double ScaledY = Offset.y + Zoom * (((double)PY / (double)Texture->Height) - 0.5);

			double x = 0;
			double y = 0;

			int Iteration = 0;

			while (x * x + y * y < 4.f && Iteration < MaxIterations)
			{
				double XTemp = x * x - y * y + ScaledX;
				y = 2 * x * y + ScaledY;
				x = XTemp;
				Iteration++;
			}


			const int Index = 4 * (Texture->Width * PY + PX);
			Texture->Data[Index] = (Uint8)((0.5f * sin(a * Iteration) + 0.5f) * 255);
			Texture->Data[Index + 1] = (Uint8)((0.5f * sin(a * Iteration + 2.094f) + 0.5f) * 255);
			Texture->Data[Index + 2] = (Uint8)((0.5f * sin(a * Iteration + 4.188f) * 255));
			Texture->Data[Index + 3] = (Uint8)255;
		}
	}

}

void OptimisedMandelbrot(ColorTexture* Texture, const Vector2f TileOffset, const Vector2f TileSize, const Vector2d Offset, const double Zoom, const int MaxIterations)
{
	const int MaxPixelX = (int)TileOffset.x + (int)TileSize.x;
	const int MaxPixelY = (int)TileOffset.y + (int)TileSize.y;

	const float a = 0.1f;

	for (int PY = (int)TileOffset.y; PY < MaxPixelY; PY++)
	{
		for (int PX = (int)TileOffset.x; PX < MaxPixelX; PX++)
		{
			double ScaledX = Offset.x + Zoom * (((double)PX / (double)Texture->Width) - 0.5);
			double ScaledY = Offset.y + Zoom * (((double)PY / (double)Texture->Height) - 0.5);

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

void OptimisedSIMDMandelbrot(ColorTexture* Texture, const Vector2f TileOffset, const Vector2f TileSize, const Vector2d Offset, const double Zoom, const int MaxIterations)
{
	const int MaxPixelX = (int)TileOffset.x + (int)TileSize.x;
	const int MaxPixelY = (int)TileOffset.y + (int)TileSize.y;

	const float a = 0.1f;

	__m256d _x, _y, _x2, _y2;

	__m256d _ConstHalf = _mm256_set1_pd(0.5);
	__m256d _ConstTwo = _mm256_set1_pd(2.0);
	__m256d _ConstFour = _mm256_set1_pd(4.0);
	__m256i _ConstOneInt = _mm256_set1_epi64x(1);

	__m256d _ConstOffset = _mm256_set1_pd(Offset.x);
	__m256d _ConstZoom = _mm256_set1_pd(Zoom);
	__m256d _ConstTextureWidth = _mm256_set1_pd(Texture->Width);

	__m256d _ScaledX, _ScaledY;

	__m256d _Mask1, _ComparisonAdd;
	__m256i _Mask2, _Temp; 
	__m256i _Iterations = _mm256_set1_epi64x(0);
	__m256i _MaxIterations = _mm256_set1_epi64x(MaxIterations);


	for (int PY = (int)TileOffset.y; PY < MaxPixelY; PY++)
	{
		_ScaledY = _mm256_set1_pd(Offset.y + Zoom * (((double)PY / (double)Texture->Height) - 0.5));

		for (int PX = (int)TileOffset.x; PX < MaxPixelX; PX +=4)
		{
			__m256d _ConstPX = _mm256_set1_pd(PX);
			_ScaledX = _mm256_set_pd(0, 1, 2, 3);
			_ScaledX = _mm256_add_pd(_ScaledX, _ConstPX);
			_ScaledX = _mm256_div_pd(_ScaledX, _ConstTextureWidth);
			_ScaledX = _mm256_sub_pd(_ScaledX, _ConstHalf);
			_ScaledX = _mm256_fmadd_pd(_ScaledX, _ConstZoom, _ConstOffset);

			//_ScaledX = _mm256_set_pd(Offset.x + Zoom * (((double)PX / (double)Texture->Width) - 0.5),
			//	Offset.x + Zoom * (((double)(PX + 1) / (double)Texture->Width) - 0.5),
			//	Offset.x + Zoom * (((double)(PX + 2) / (double)Texture->Width) - 0.5),
			//	Offset.x + Zoom * (((double)(PX + 3) / (double)Texture->Width) - 0.5));			

			_x = _mm256_setzero_pd();
			_y = _mm256_setzero_pd();
			_x2 = _mm256_setzero_pd();
			_y2 = _mm256_setzero_pd();
			_Iterations = _mm256_setzero_si256();
			
			//We do this hack cause we want ot keep the vars in scope, definign them potuside is ugly (may apparently initalise them too?)
			//Inner loop
			repeat:
			_x = _mm256_add_pd(_x, _x);
			_y = _mm256_fmadd_pd(_x, _y, _ScaledY);
			_x = _mm256_sub_pd(_x2, _y2);
			_x = _mm256_add_pd(_x, _ScaledX);
			_x2 = _mm256_mul_pd(_x, _x);
			_y2 = _mm256_mul_pd(_y, _y);

			//Comparison
			_ComparisonAdd = _mm256_add_pd(_x2, _y2);

			//sets all bits to one for each 64 bit if true
			_Mask1 = _mm256_cmp_pd(_ComparisonAdd, _ConstFour, _CMP_LT_OQ);
			_Mask2 = _mm256_cmpgt_epi64(_MaxIterations, _Iterations);

			//ands then because we want both statment to be true
			_Mask2 = _mm256_and_si256(_Mask2, _mm256_castpd_si256(_Mask1));
			
			//We only want to add iterations to values satisfyign IF condition
			_Temp = _mm256_and_si256(_ConstOneInt, _Mask2);
			//add the 1's
			_Iterations = _mm256_add_epi64(_Iterations, _Temp);
								
			//Converts register to one big integer, if bigger than zero means we have at leasat one continuing value
			if (_mm256_movemask_pd(_mm256_castsi256_pd(_Mask2)) > 0) goto repeat;

			//Extracting data
			const int Index = 4 * (Texture->Width * PY + PX);

			Texture->Data[Index] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[3]) + 0.5f) * 255);
			Texture->Data[Index + 1] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[3] + 2.094f) + 0.5f) * 255);
			Texture->Data[Index + 2] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[3] + 4.188f) * 255));
			Texture->Data[Index + 3] = (Uint8)255;

			if (PX + 1 < MaxPixelX)
			{
				Texture->Data[Index + 4] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2]) + 0.5f) * 255);
				Texture->Data[Index + 5] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 6] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2] + 4.188f) * 255));
				Texture->Data[Index + 7] = (Uint8)255;
			}

			if (PX + 2 < MaxPixelX)
			{
				Texture->Data[Index + 8] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1]) + 0.5f) * 255);
				Texture->Data[Index + 9] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 10] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1] + 4.188f) * 255));
				Texture->Data[Index + 11] = (Uint8)255;
			}

			if (PX + 3 < MaxPixelX)
			{
				Texture->Data[Index + 12] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0]) + 0.5f) * 255);
				Texture->Data[Index + 13] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 14] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 4.188f) * 255));
				Texture->Data[Index + 15] = (Uint8)255;
			}
		}
	}
}

/*
void OptimisedSIMD512Mandelbrot(ColorTexture* Texture, const Vector2f TileOffset, const Vector2f TileSize, const Vector2d Offset, const double Zoom, const int MaxIterations)
{
	float MandelbrotX;
	float MandelbrotY;

	int MaxPixelX = (int)TileOffset.x + (int)TileSize.x;
	int MaxPixelY = (int)TileOffset.y + (int)TileSize.y;

	__m512d _x, _y, _x2, _y2;

	__m512d _ConstTwo = _mm512_set1_pd(2.0);
	__m512d _ConstFour = _mm512_set1_pd(4.0);
	__m512d _ScaledX, _ScaledY;

	__m512d _ComparisonAdd;
	__mmask8 _Mask1, _Mask2, _Temp;
	__m512i _ConstOneInt = _mm512_set1_epi64(1);
	__m512i _Iterations = _mm512_set1_epi64(0);
	__m512i _MaxIterations = _mm512_set1_epi64(MaxIterations);

	float a = 0.1f;

	for (int PY = (int)TileOffset.y; PY < MaxPixelY; PY++)
	{
		_ScaledY = _mm512_set_pd(Offset.y + ((double)PY / (double)Text.Height) * Zoom);

		for (int PX = (int)TileOffset.x; PX < MaxPixelX; PX += 8)
		{
			_ScaledX = _mm512_set_pd(Offset.x + ((double)PX / (double)Text.Width) * Zoom,
				Offset.x + ((double)(PX + 1) / (double)Text.Width) * Zoom,
				Offset.x + ((double)(PX + 2) / (double)Text.Width) * Zoom,
				Offset.x + ((double)(PX + 3) / (double)Text.Width) * Zoom,
				Offset.x + ((double)(PX + 4) / (double)Text.Width) * Zoom,
				Offset.x + ((double)(PX + 5) / (double)Text.Width) * Zoom,
				Offset.x + ((double)(PX + 6) / (double)Text.Width) * Zoom,
				Offset.x + ((double)(PX + 7) / (double)Text.Width) * Zoom);

			_x = _mm512_setzero_pd();
			_y = _mm512_setzero_pd();
			_x2 = _mm512_setzero_pd();
			_y2 = _mm512_setzero_pd();
			_Iterations = _mm512_setzero_si512();

			//inner loop
		repeat:
			_x = _mm512_add_pd(_x, _x);
			_y = _mm512_fmadd_pd(_x, _y, _ScaledY);
			_x = _mm512_sub_pd(_x2, _y2);
			_x = _mm512_add_pd(_x, _ScaledX);
			_x2 = _mm512_mul_pd(_x, _x);
			_y2 = _mm512_mul_pd(_y, _y);

			//Comparison
			_ComparisonAdd = _mm512_add_pd(_x2, _y2);

			//sets all bits to one for each 64 bit if true
			_Mask1 = _mm512_cmp_pd_mask(_ComparisonAdd, _ConstFour, _CMP_LT_OQ);
			_Mask2 = _mm512_cmpgt_epi64_mask(_MaxIterations, _Iterations);

			//ands then because we want both statment to be true
			_Mask2 = _mm512_and_si512(_Mask2, _mm512_castpd_si512(_Mask1));

			//We only want to add iterations to values satisfyign IF condition
			_Temp = _mm512_and_si512(_ConstOneInt, _Mask2);
			//add the 1's
			_Iterations = _mm256_add_epi64(_Iterations, _Temp);

			//converts register to one big integer, if bigger than zero means we have at leasat one continuing value
			if (_mm256_movemask_pd(_mm256_castsi256_pd(_Mask2)) > 0) goto repeat;

			//Extracting data
			const int Index = 4 * (Texture->Width * PY + PX);

			Texture->Data[Index] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[3]) + 0.5f) * 255);
			Texture->Data[Index + 1] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[3] + 2.094f) + 0.5f) * 255);
			Texture->Data[Index + 2] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[3] + 4.188f) * 255));
			Texture->Data[Index + 3] = (Uint8)255;

			if (PX + 1 < MaxPixelX)
			{
				Texture->Data[Index + 4] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2]) + 0.5f) * 255);
				Texture->Data[Index + 5] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 6] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2] + 4.188f) * 255));
				Texture->Data[Index + 7] = (Uint8)255;
			}

			if (PX + 2 < MaxPixelX)
			{
				Texture->Data[Index + 8] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1]) + 0.5f) * 255);
				Texture->Data[Index + 9] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 10] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1] + 4.188f) * 255));
				Texture->Data[Index + 11] = (Uint8)255;
			}

			if (PX + 3 < MaxPixelX)
			{
				Texture->Data[Index + 12] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0]) + 0.5f) * 255);
				Texture->Data[Index + 13] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 14] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 4.188f) * 255));
				Texture->Data[Index + 15] = (Uint8)255;
			}
			
			if (PX + 4 < MaxPixelX)
			{
				Texture->Data[Index + 16] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2]) + 0.5f) * 255);
				Texture->Data[Index + 17] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 18] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[2] + 4.188f) * 255));
				Texture->Data[Index + 19] = (Uint8)255;
			}

			if (PX + 5 < MaxPixelX)
			{
				Texture->Data[Index + 20] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1]) + 0.5f) * 255);
				Texture->Data[Index + 21] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 22] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[1] + 4.188f) * 255));
				Texture->Data[Index + 23] = (Uint8)255;
			}

			if (PX + 6 < MaxPixelX)
			{
				Texture->Data[Index + 24] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0]) + 0.5f) * 255);
				Texture->Data[Index + 25] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 26] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 4.188f) * 255));
				Texture->Data[Index + 27] = (Uint8)255;
			}

			if (PX + 7 < MaxPixelX)
			{
				Texture->Data[Index + 28] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0]) + 0.5f) * 255);
				Texture->Data[Index + 29] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 2.094f) + 0.5f) * 255);
				Texture->Data[Index + 30] = (Uint8)((0.5f * sin(a * _Iterations.m256i_i64[0] + 4.188f) * 255));
				Texture->Data[Index + 31] = (Uint8)255;
			}

		}
	}
}
*/