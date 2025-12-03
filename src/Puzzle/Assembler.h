#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "../Features/FeatureExtractor.h"
#include "../Matching/PieceMatcher.h"

namespace Assembler {

    cv::Mat assembleImage(const PuzzleLayout& layout,
                          const std::vector<PieceFeature>& features,
                          int canvasW,
                          int canvasH);

    std::vector<cv::Point> computePiecePositions(
        const PuzzleLayout& layout,
        const std::vector<PieceFeature>& features,
        int canvasW,
        int canvasH
    );
}