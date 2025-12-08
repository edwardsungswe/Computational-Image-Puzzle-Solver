#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "../Features/FeatureExtractor.h"
#include "../Matching/PieceMatcher.h"

namespace Assembler {
    cv::Mat assemble(const std::vector<PieceFeature>& features, const PuzzleLayout& layout);
    cv::Mat assemble(const std::vector<PieceFeature>& features, const PuzzleLayout& layout,
                     int outputWidth, int outputHeight);
    cv::Mat assembleWithDebug(const std::vector<PieceFeature>& features, const PuzzleLayout& layout);
}