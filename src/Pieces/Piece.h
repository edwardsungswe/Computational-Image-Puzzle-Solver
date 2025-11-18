
#pragma once
#include <opencv2/opencv.hpp>

struct Piece {
    cv::Mat image;
    cv::RotatedRect boundingBox;
    int id;
};
