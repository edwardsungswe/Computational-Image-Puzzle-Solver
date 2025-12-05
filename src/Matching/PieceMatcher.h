#pragma once
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "../Features/FeatureExtractor.h"


struct PiecePosition {
    cv::Point2f position;
    float rotation;
    cv::Size size;
};

struct PuzzleLayout {
    std::vector<std::vector<int>> grid;
    std::unordered_map<int, PiecePosition> positions;
    cv::Rect2f bounds;
    int rows, cols;
};


struct Pair {
    int pieceA, pieceB;
    int edgeA, edgeB;
    double val;
};


namespace PieceMatcher {
    std::vector<Pair> createFilteredMatches(const std::vector<PieceFeature>& features, double ratioTestThreshold);
    PuzzleLayout buildLayout(const std::vector<Pair>& matches, const std::vector<PieceFeature>& f, int canvasW, int canvasH);
    cv::Mat rotatePiece(const cv::Mat& img, float rotation);
    PuzzleLayout buildLayoutRasterScan(const std::vector<PieceFeature>& features, int canvasW, int canvasH);

}