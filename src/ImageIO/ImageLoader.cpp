#include "ImageLoader.h"
#include <fstream>
#include <iostream>
#include <vector>

cv::Mat ImageLoader::loadPNG(const std::string& path) {
    std::cout << "Detec567567tasded ";
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error loading PNG: " << path << std::endl;
    }
    return img;
}

unsigned char* ImageLoader::loadRawRGB(const std::string& path, int width, int height) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error Opening File for Reading: " << path << std::endl;
        return nullptr;
    }

    std::vector<char> Rbuf(width * height);
    std::vector<char> Gbuf(width * height);
    std::vector<char> Bbuf(width * height);

    file.read(Rbuf.data(), width * height);
    file.read(Gbuf.data(), width * height);
    file.read(Bbuf.data(), width * height);

    unsigned char* data = (unsigned char*)malloc(width * height * 3);

    for (int i = 0; i < width * height; i++) {
        data[3 * i]     = Rbuf[i];
        data[3 * i + 1] = Gbuf[i];
        data[3 * i + 2] = Bbuf[i];
    }

    return data;
}

