#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct EdgeFeature {
    std::vector<cv::Vec3b> boundaryPixels;  // Actual pixels along the edge
    std::vector<double> gradients;           // Gradient values along edge
    int length;                              // Edge length in pixels
};

struct PieceFeature {
    cv::Mat img;
    EdgeFeature top, right, bottom, left;
    cv::Size size;
};

namespace FeatureExtractor {
    PieceFeature extract(const cv::Mat& piece);
    EdgeFeature extractEdge(const cv::Mat& img, int edge);
}