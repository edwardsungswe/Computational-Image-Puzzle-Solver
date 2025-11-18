#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ImageLoader {
public:
    static cv::Mat loadPNG(const std::string& path);
    static unsigned char* loadRawRGB(const std::string& path, int width, int height);
};
