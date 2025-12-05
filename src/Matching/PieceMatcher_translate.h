#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "../Features/FeatureExtractor.h"
#include "PieceMatcher.h"

namespace PieceMatcher_translate {

struct Pair {
    int pieceA;
    int pieceB;
    int edgeA; // 0 top, 1 right, 2 bottom, 3 left
    int edgeB;
    double val;
};

std::vector<Pair> createFilteredMatches(
        const std::vector<PieceFeature>& features,
        double ratioTestThreshold = 0.8);

PuzzleLayout buildLayout(
        const std::vector<Pair>& matches,
        const std::vector<PieceFeature>& f,
        int canvasW,
        int canvasH);

}
