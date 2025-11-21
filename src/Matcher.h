#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "FeatureExtractor.h"

// struct PuzzleLayout {
//     std::vector<std::vector<int>> grid;
// };

struct PuzzleLayout {
    std::unordered_map<int, PiecePosition> position;
    cv::Rect2f bounds;
}

struct PiecePosition {
    cv::Point2f position;
    float rotation;
    cv::Size size;
}

struct Pair {
    int pieceA, pieceB;
    int edgeA, edgeB;
    double val;
};


namespace Matcher {
    std::vector<std::pair<int,double>> matchAll(const std::vector<PieceFeature>& features);
    PuzzleLayout buildLayout(int pieceCount);


    private:
    
}
