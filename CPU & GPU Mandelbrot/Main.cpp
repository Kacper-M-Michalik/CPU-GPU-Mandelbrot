#include <thread>
#include <chrono>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <Mandelbrot.h>
#include <GPUMandelbrot.h>
#include <PerformanceCounter.h>

//FIX CONSOLE ISSUE LATER

int main()
{
    unsigned int WindowWidth = 1200;
    unsigned int WindowHeight = 800;

    unsigned int RenderAreaX = 1200;
    unsigned int RenderAreaY = 800;

    sf::RenderWindow window(sf::VideoMode(WindowWidth, WindowHeight), "Mandelbrot");


    //sf::RenderTexture GPUTexture;
    GPUColorTexture* GPUProcessingTexture = Setup(RenderAreaX, RenderAreaY);
    sf::Texture GPUTexture;
    GPUTexture.create(GPUProcessingTexture->Width, GPUProcessingTexture->Height);
    GPUTexture.setSmooth(false);
    GPUTexture.setRepeated(false);

    sf::Texture CPUTexture;
    CPUTexture.create(RenderAreaX, RenderAreaY);
    CPUTexture.setSmooth(false);
    CPUTexture.setRepeated(false);
    ColorTexture CPUProcessingTexture(RenderAreaX, RenderAreaY);
   
    int RenderMode = 0;
    int FrameMeasureCount = 100;
    int CurrentFrame = 0;
    PerformanceCounter CPUCounter(FrameMeasureCount);
    PerformanceCounter GPUCounter1(FrameMeasureCount);
    PerformanceCounter GPUCounter2(FrameMeasureCount);

    sf::Vector2f TileSize(RenderAreaX, (int)ceil((float)RenderAreaY / (float)std::thread::hardware_concurrency()));
    std::vector<std::thread> Threads;
    sf::Vector2d Offset(0, 0);    
    double Zoom = 1;
    int Iterations = 64;

    sf::Sprite MandelbrotSprite;
    sf::Vector2i PreviousPosition = sf::Mouse::getPosition(window);

    while (window.isOpen())
    {
        CurrentFrame++;

        sf::Event event;
        while (window.pollEvent(event))
        {           
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }

            if (event.type == sf::Event::MouseWheelScrolled && event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
            {
                Zoom = Zoom * (1 - event.mouseWheelScroll.delta * 0.1f);
            }

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Left)
            {
                Iterations += -32;
            }
            
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Right)
            {
                Iterations += 32;
            }

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space)
            {
                RenderMode++;
                if (RenderMode > 2) RenderMode = 0;
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

        auto StartTime = std::chrono::high_resolution_clock::now();
        if (RenderMode == 0)
        {
            int XTiles = ceil((float)RenderAreaX / (float)TileSize.x);
            int YTiles = ceil((float)RenderAreaY / (float)TileSize.y);

            for (int x = 0; x < RenderAreaX; x += TileSize.x)
            {
                for (int y = 0; y < RenderAreaY; y += TileSize.y)
                {
                    sf::Vector2f ActualTileSize(TileSize);

                    if (RenderAreaX - x < TileSize.x)
                    {
                        ActualTileSize.x = RenderAreaX % (int)TileSize.x;
                    }
                    if (RenderAreaY - y < TileSize.y)
                    {
                        ActualTileSize.y = RenderAreaY % (int)TileSize.y;
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
            MandelbrotSprite.setTexture(CPUTexture, true);
        }
        else if (RenderMode == 1)
        {
            RunGPUMandelbrot(GPUProcessingTexture, 0, Offset, Zoom, Iterations);
            GPUTexture.update(GPUProcessingTexture->CPUTexture);
            MandelbrotSprite.setTexture(GPUTexture, true);
        }
        else 
        {
            RunGPUMandelbrot(GPUProcessingTexture, 1, Offset, Zoom, Iterations);
            GPUTexture.update(GPUProcessingTexture->CPUTexture);
            MandelbrotSprite.setTexture(GPUTexture, true);
        }

        double TimeMS = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - StartTime).count();

        if (RenderMode == 0) 
        {
            CPUCounter.AddTime(TimeMS);
            if (CurrentFrame >= FrameMeasureCount)
            {
                std::cout << CPUCounter.GetAverageTime() << std::endl;
                CurrentFrame = 0;
            }
        }
        else if (RenderMode == 1)
        {
            GPUCounter1.AddTime(TimeMS);
            if (CurrentFrame >= FrameMeasureCount)
            {
                std::cout << GPUCounter1.GetAverageTime() << std::endl;
                CurrentFrame = 0;
            }
        }
        else if (RenderMode == 2)
        {
            GPUCounter2.AddTime(TimeMS);
            if (CurrentFrame >= FrameMeasureCount)
            {
                std::cout << GPUCounter2.GetAverageTime() << std::endl;
                CurrentFrame = 0;
            }
        }

        window.clear(sf::Color::Black);

        window.draw(MandelbrotSprite);

        window.display();
    }

    return 0;
}