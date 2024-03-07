#include <thread>
#include <chrono>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <Mandelbrot.h>
#include <GPUMandelbrot.h>

//FIX CONSOLE ISSUE LATER

int main()
{
    unsigned int WindowWidth = 1000;
    unsigned int WindowHeight = 750;

    sf::RenderWindow window(sf::VideoMode(WindowWidth, WindowHeight), "Mandelbrot");

    sf::RenderTexture GPUTexture;
    GPUTexture.create(WindowWidth, WindowHeight);
    GPUTexture.setSmooth(false);
    GPUTexture.setRepeated(true);

    sf::Texture CPUTexture;
    CPUTexture.create(WindowWidth, WindowHeight);
    CPUTexture.setSmooth(false);
    CPUTexture.setRepeated(true);
    ColorTexture CPUProcessingTexture(WindowWidth, WindowHeight);
   
    bool CPUMode = true;
    sf::Sprite MandelbrotSprite;

    sf::Vector2d Offset(0, 0);    
    double Zoom = 1;
    int Iterations = 64;
    sf::Vector2f TileSize(WindowWidth, (int)ceil((float)WindowHeight / (float)std::thread::hardware_concurrency()));
    std::vector<std::thread> Threads;

    sf::Vector2i PreviousPosition = sf::Mouse::getPosition(window);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {           
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }

            if (event.type == sf::Event::MouseWheelScrolled && event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
            {
                Zoom += -event.mouseWheelScroll.delta * 0.1;
            }
        } 
        
        sf::Vector2i NewPosition = sf::Mouse::getPosition(window);
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        {
            sf::Vector2i Delta = NewPosition - PreviousPosition;
            Offset.x -= Zoom * (double)Delta.x / WindowWidth;
            Offset.y -= Zoom * (double)Delta.y / WindowHeight;
        }
        PreviousPosition = NewPosition;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
        {
            CPUMode = !CPUMode;
        }

        auto StartTime = std::chrono::high_resolution_clock::now();
        if (CPUMode)
        {
            int XTiles = ceil((float)WindowWidth / (float)TileSize.x);
            int YTiles = ceil((float)WindowHeight / (float)TileSize.y);

            for (int x = 0; x < WindowWidth; x += TileSize.x)
            {
                for (int y = 0; y < WindowHeight; y += TileSize.y)
                {
                    sf::Vector2f ActualTileSize(TileSize);

                    if (WindowWidth - x < TileSize.x)
                    {
                        ActualTileSize.x = WindowWidth % (int)TileSize.x;
                    }
                    if (WindowHeight - y < TileSize.y)
                    {
                        ActualTileSize.y = WindowHeight % (int)TileSize.y;
                    }
                    

                    Threads.push_back(std::thread(OptimisedSIMDMandelbrot, &CPUProcessingTexture, sf::Vector2f(x, y), ActualTileSize, Offset, Zoom, Iterations));
                }
            }

            for (int i = 0; i < Threads.size(); i++)
            {
                Threads[i].join();
            }
            Threads.clear();

            CPUTexture.update(CPUProcessingTexture.Data);
            MandelbrotSprite.setTexture(CPUTexture);
        }
        else
        {
            MandelbrotSprite.setTexture(GPUTexture.getTexture());
        }
        double TimeMS = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - StartTime).count();

        std::cout << TimeMS << std::endl;

        window.clear(sf::Color::Black);

        window.draw(MandelbrotSprite);

        window.display();
    }

    return 0;
}