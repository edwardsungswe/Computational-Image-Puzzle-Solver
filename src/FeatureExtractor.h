#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct EdgeFeature {
    std::vector<double> vals;
};

struct PieceFeature {
    cv::Mat img;
    EdgeFeature top, right, bottom, left;
    std::vector<cv::KeyPoint> keypoints[4];
    cv::Mat descriptors[4];
};


namespace FeatureExtractor {
    PieceFeature extract(const cv::Mat& piece);
}