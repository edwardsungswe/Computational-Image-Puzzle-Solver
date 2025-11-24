#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "../FeatureExtractor.h"
#include "../Matcher.h"

namespace PuzzleAnimator {

    /**
     * Represents the state of a piece during animation
     */
    struct AnimationFrame {
        cv::Point2f position;      // Current position
        float rotation;             // Current rotation angle in degrees
        float scale;                // Current scale factor
    };

    /**
     * Configuration for animation playback
     */
    struct AnimationConfig {
        int totalFrames = 120;      // Total frames (at 30fps = 4 seconds)
        int fps = 30;               // Frames per second
        bool saveFrames = false;    // Whether to save frames to disk
        std::string outputDir = "./animation_frames/";
        bool showWindow = true;     // Whether to display animation in window
    };

    /**
     * Animates puzzle piece assembly from scrambled state to final position
     * @param features Vector of piece features
     * @param layout The final puzzle layout with piece positions and rotations
     * @param canvasW Canvas width
     * @param canvasH Canvas height
     * @param config Animation configuration
     */
    void animatePuzzleAssembly(
        const std::vector<PieceFeature>& features,
        const PuzzleLayout& layout,
        int canvasW,
        int canvasH,
        const AnimationConfig& config = AnimationConfig()
    );

    /**
     * Compute intermediate frame state for a single piece
     * @param startPos Initial position
     * @param endPos Final position
     * @param startRot Initial rotation angle
     * @param endRot Final rotation angle
     * @param t Time parameter [0, 1]
     * @return AnimationFrame with interpolated values
     */
    AnimationFrame interpolateFrame(
        const cv::Point2f& startPos,
        const cv::Point2f& endPos,
        float startRot,
        float endRot,
        float t
    );

}
