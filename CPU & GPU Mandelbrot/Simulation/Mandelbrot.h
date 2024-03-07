#pragma once
#include <SFML/Graphics.hpp>

struct ColorTexture
{
	unsigned int Width;
	unsigned int Height;

	sf::Uint8* Data;

	ColorTexture(unsigned int SetWidth, unsigned int SetHeight)
	{
		Width = SetWidth;
		Height = SetHeight;

		Data = (sf::Uint8*)malloc(sizeof(sf::Uint8) * Width * Height * 4);
	}

	//No clue why but passing in a ColoxTexture reference isntead of pointer causes the ColorTexture to get destoryed and the destructor called
	~ColorTexture() 
	{
		free(Data);
	}
};

void SimpleMandelbrot(ColorTexture* Texture, const sf::Vector2f TileOffset, const sf::Vector2f TileSize, const sf::Vector2d Offset, const double Zoom, const int MaxIterations);
void OptimisedMandelbrot(ColorTexture* Text, const sf::Vector2f TileOffset, const sf::Vector2f TileSize, const sf::Vector2d Offset, const double Zoom, const int MaxIterations);
void OptimisedSIMDMandelbrot(ColorTexture* Text, const sf::Vector2f TileOffset, const sf::Vector2f TileSize, const sf::Vector2d Offset, const double Zoom, const int MaxIterations);