#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "../Features/FeatureExtractor.h"
#include "../Matching/PieceMatcher.h"

namespace PuzzleAnimator {

    struct AnimationFrame {
        cv::Point2f position;
        float rotation;
        float scale;
    };

    struct AnimationConfig {
        int totalFrames = 120;
        int fps = 30;
        bool saveFrames = false;
        std::string outputDir = "./animation_frames/";
        bool showWindow = true;
    };

    void animatePuzzleAssembly(
        const std::vector<PieceFeature>& features,
        const PuzzleLayout& layout,
        int canvasW,
        int canvasH,
        const std::vector<cv::Point2f>& initialPositions,
        const std::vector<float>& initialRotations,
        const AnimationConfig& config = AnimationConfig()
    );

    void animatePuzzleAssembly(
        const std::vector<PieceFeature>& features,
        const PuzzleLayout& layout,
        int canvasW,
        int canvasH,
        const AnimationConfig& config = AnimationConfig()
    );

    AnimationFrame interpolateFrame(
        const cv::Point2f& startPos,
        const cv::Point2f& endPos,
        float startRot,
        float endRot,
        float t
    );

}