#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class PieceExtractor {
public:
    // optional: save pieces to directory. If outputDir is not empty, each piece will be saved as a PNG file to the directory.
    static std::vector<cv::Mat> extractPieces(const cv::Mat& rgbImage, bool enableRotation, 
                                               const std::string& outputDir = "");
    static cv::Mat drawContoursOnImage(const cv::Mat& rgbImage);
};
