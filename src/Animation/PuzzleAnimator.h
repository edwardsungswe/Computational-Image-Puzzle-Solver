#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "../Features/FeatureExtractor.h"
#include "../Matching/PieceMatcher.h"

namespace PuzzleAnimator {

    
    //Represents the state of a piece during animation
    
    struct AnimationFrame {
        cv::Point2f position;      // Current position
        float rotation;             // Current rotation angle in degrees
        float scale;                // Current scale factor
    };

    
    //Configuration for animation playback
    
    struct AnimationConfig {
        int totalFrames = 120;      // Total frames (at 30fps = 4 seconds)
        int fps = 30;               // Frames per second
        bool saveFrames = false;    // Whether to save frames to disk
        std::string outputDir = "./animation_frames/";
        bool showWindow = true;     // Whether to display animation in window
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
