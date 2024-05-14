#include <thread>
#include <chrono>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <Mandelbrot.h>
#include <GPUMandelbrot.h>
#include <PerformanceCounter.h>


int main()
{
    unsigned int WindowWidth = 1200;
    unsigned int WindowHeight = 900;
    sf::RenderWindow window(sf::VideoMode(WindowWidth, WindowHeight), "Mandelbrot/Julia Grapher");


    //By default present a 2 unit height (in mdbrt coords) view on the mandelbrot set
    int RenderAreaX = WindowWidth;
    int RenderAreaY = WindowHeight;
    const sf::Vector2d PixelsPerUnit(WindowHeight / 2.0, WindowHeight / 2.0);

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
    PerformanceCounter CPUCounter(FrameMeasureCount, "CPU Mandelbrot");
    PerformanceCounter GPUCounter1(FrameMeasureCount, "GPU Mandelbrot");
    PerformanceCounter GPUCounter2(FrameMeasureCount, "GPU Julia");
    PerformanceCounter* CurrentCounter = &CPUCounter;

    sf::Vector2f TileSize(RenderAreaX, (int)ceil((float)RenderAreaY / (float)std::thread::hardware_concurrency()));
    std::vector<std::thread> Threads;


    sf::Vector2d MandelbrotOffset(0, 0);
    double Zoom = 1;

    sf::Vector2d JuliaOffset(0, 0);
    double JuliaZoom = 1;
    sf::Vector2d C(-0.162, 1.04);
    // x: 0.408889, y: -0.34
    // x: -0.767715, y: 0.105779
    // x:-0.77146, y: -0.10119
    int Iterations = 128;


    sf::Sprite MandelbrotSprite; 
    MandelbrotSprite.setTexture(CPUTexture, false);
    MandelbrotSprite.setTextureRect(sf::IntRect(0, RenderAreaY, RenderAreaX, -RenderAreaY));
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

            if (event.type == sf::Event::MouseWheelScrolled && event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel && RenderMode <= 1)
            {
                Zoom = Zoom * (1 - event.mouseWheelScroll.delta * 0.1f);
            }

            if (event.type == sf::Event::MouseWheelScrolled && event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel && RenderMode >= 2)
            {
                JuliaZoom = JuliaZoom * (1 - event.mouseWheelScroll.delta * 0.1f);
            }

            if (event.type == sf::Event::KeyPressed && (event.key.code == sf::Keyboard::Right || event.key.code == sf::Keyboard::W))
            {
                Iterations += 32;
            }

            if (event.type == sf::Event::KeyPressed && (event.key.code == sf::Keyboard::Left || event.key.code == sf::Keyboard::S) && Iterations > 32)
            {
                Iterations += -32;
            }

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space)
            {
                RenderMode++;
                if (RenderMode > 2) RenderMode = 0;

                if (RenderMode == 0)
                {
                    MandelbrotSprite.setTexture(CPUTexture, false);
                    MandelbrotSprite.setTextureRect(sf::IntRect(0, RenderAreaY, RenderAreaX, -RenderAreaY));
                    CurrentCounter = &CPUCounter;
                }
                else if (RenderMode == 1)
                {
                    MandelbrotSprite.setTexture(GPUTexture, false);
                    MandelbrotSprite.setTextureRect(sf::IntRect((GPUProcessingTexture->Width - RenderAreaX) / 2, GPUProcessingTexture->Height - (GPUProcessingTexture->Height - RenderAreaY) / 2, RenderAreaX, -RenderAreaY));
                    CurrentCounter = &GPUCounter1;
                }
                else if (RenderMode == 2)
                {
                    MandelbrotSprite.setTexture(GPUTexture, false);
                    MandelbrotSprite.setTextureRect(sf::IntRect((GPUProcessingTexture->Width - RenderAreaX) / 2, GPUProcessingTexture->Height - (GPUProcessingTexture->Height - RenderAreaY) / 2, RenderAreaX, -RenderAreaY));
                    CurrentCounter = &GPUCounter2;
                }
            }
        } 
        

        sf::Vector2i NewPosition = sf::Mouse::getPosition(window);
        sf::Vector2i Delta = NewPosition - PreviousPosition;

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && RenderMode <= 1)
        {
            MandelbrotOffset.x -= Zoom * (double)Delta.x / PixelsPerUnit.x;
            MandelbrotOffset.y += Zoom * (double)Delta.y / PixelsPerUnit.y;
        }
        //if (sf::Mouse::isButtonPressed(sf::Mouse::Right) && RenderMode <= 1)
        if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
        {
            C.x = MandelbrotOffset.x + Zoom * (((double)NewPosition.x - (double)WindowWidth / 2) / PixelsPerUnit.x);
            C.y = MandelbrotOffset.y + Zoom * (((double)WindowHeight / 2 - (double)NewPosition.y) / PixelsPerUnit.y);
        }
        
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && RenderMode >= 2)
        {
            JuliaOffset.x -= JuliaZoom * (double)Delta.x / PixelsPerUnit.x;
            JuliaOffset.y += JuliaZoom * (double)Delta.y / PixelsPerUnit.y;
        }

        PreviousPosition = NewPosition;


        auto StartTime = std::chrono::high_resolution_clock::now();
        if (RenderMode == 0)
        {
            sf::Vector2d DrawArea = sf::Vector2d(Zoom * ((double)RenderAreaX / PixelsPerUnit.x), Zoom * ((double)RenderAreaY / PixelsPerUnit.y));

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

                    Threads.push_back(std::thread(OptimisedSIMDMandelbrot, &CPUProcessingTexture, sf::Vector2f(x, y), ActualTileSize, MandelbrotOffset, DrawArea, Iterations));
                }
            }

            for (int i = 0; i < Threads.size(); i++)
            {
                Threads[i].join();
            }
            Threads.clear();

            CPUTexture.update(CPUProcessingTexture.Data);
        }
        else if (RenderMode == 1)
        {
            RunMandelbrotKernel(GPUProcessingTexture, MandelbrotOffset, sf::Vector2d(Zoom * ((double)RenderAreaX / PixelsPerUnit.x) * ((double)GPUProcessingTexture->Width / (double)RenderAreaX), Zoom * ((double)RenderAreaY / PixelsPerUnit.y) * ((double)GPUProcessingTexture->Height / (double)RenderAreaY)), Iterations);
            GPUTexture.update(GPUProcessingTexture->CPUTexture);
        }
        else if (RenderMode == 2)
        {
            RunJuliaKernel(GPUProcessingTexture, JuliaOffset, sf::Vector2d(JuliaZoom* ((double)RenderAreaX / PixelsPerUnit.x) * ((double)GPUProcessingTexture->Width / (double)RenderAreaX), JuliaZoom * ((double)RenderAreaY / PixelsPerUnit.y) * ((double)GPUProcessingTexture->Height / (double)RenderAreaY)), C, Iterations);
            GPUTexture.update(GPUProcessingTexture->CPUTexture);
        }

        window.clear(sf::Color::Black);
        window.draw(MandelbrotSprite);
        window.display();

        double TimeMS = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - StartTime).count();      
        CurrentCounter->AddTime(TimeMS);
        if (CurrentFrame >= FrameMeasureCount)
        {
            std::cout << "x: " << C.x << std::endl;
            std::cout << "y: " << C.y << std::endl;
            std::cout << CurrentCounter->Name << std::endl;
            std::cout << CurrentCounter->GetAverageTime() << std::endl << std::endl;
            CurrentFrame = 0;
        }
    }

    return 0;
}