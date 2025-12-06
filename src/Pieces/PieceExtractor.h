#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct PieceInfo {
    cv::Mat img;              // The extracted piece image
    cv::Point2f center;       // Center position in original image
    float originalRotation;   // Rotation angle from original image
    cv::Size originalSize;    // Size before extraction
};

class PieceExtractor {
public:
    // optional: save pieces to directory. If outputDir is not empty, each piece will be saved as a PNG file to the directory.
    static std::vector<cv::Mat> extractPieces(const cv::Mat& rgbImage, bool enableRotation, 
                                               const std::string& outputDir = "");
    
    // New method that returns pieces with position/rotation info for animation
    static std::vector<PieceInfo> extractPiecesWithInfo(const cv::Mat& rgbImage, bool enableRotation, 
                                                        const std::string& outputDir = "");
    
    static cv::Mat drawContoursOnImage(const cv::Mat& rgbImage);
};
