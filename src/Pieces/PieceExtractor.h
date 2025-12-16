#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct PieceInfo {
    cv::Mat img;
    cv::Point2f center;
    float originalRotation;
    cv::Size originalSize;
};

class PieceExtractor {
public:
    static std::vector<cv::Mat> extractPieces(const cv::Mat& bgrImage, const std::string& outputDir = "");
    static std::vector<PieceInfo> extractPiecesWithInfo(const cv::Mat& bgrImage, const std::string& outputDir = "");
    static cv::Mat drawContoursOnImage(const cv::Mat& bgrImage);
};