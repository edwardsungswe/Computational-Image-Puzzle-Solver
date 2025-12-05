#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PieceExtractor {
public:
    static std::vector<cv::Mat> extractPieces(const cv::Mat& rgbImage, bool enableRotation);
    static cv::Mat drawContoursOnImage(const cv::Mat& rgbImage);
};
