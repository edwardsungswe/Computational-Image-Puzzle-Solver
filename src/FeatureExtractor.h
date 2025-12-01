#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct EdgeFeature {
    std::vector<double> vals;
};

struct PieceFeature {
    cv::Mat img;
    EdgeFeature top, right, bottom, left;

};


namespace FeatureExtractor {
    PieceFeature extract(const cv::Mat& piece);
}