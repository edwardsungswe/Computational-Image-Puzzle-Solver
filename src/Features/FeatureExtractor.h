#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct EdgeFeature {
    // rgb histogram
    // [R_bin0, R_bin1, ..., R_bin7, G_bin0, G_bin1, ..., G_bin7, B_bin0, B_bin1, ..., B_bin7]
    std::vector<double> histogram;
    
    // backward compatibility: keep old sampling point way
    std::vector<double> vals;  // luminance value
    std::vector<cv::Vec3d> rgbVals;  // RGB three channels value
};

struct PieceFeature {
    cv::Mat img;
    EdgeFeature top, right, bottom, left;
};


namespace FeatureExtractor {
    PieceFeature extract(const cv::Mat& piece);
}