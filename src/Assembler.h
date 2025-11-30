#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "FeatureExtractor.h"
#include "Matcher.h"

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