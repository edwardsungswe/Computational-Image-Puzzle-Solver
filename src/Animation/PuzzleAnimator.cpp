#include "PuzzleAnimator.h"
#include "../Matcher.h"
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

namespace PuzzleAnimator {

    /**
     * Easing function for smooth animation (ease-in-out cubic)
     */
    float easeInOutCubic(float t) {
        return t < 0.5f ? 4.0f * t * t * t : 1.0f - pow(-2.0f * t + 2.0f, 3) / 2.0f;
    }

    /**
     * Rotate a point around origin
     */
    cv::Point2f rotatePoint(const cv::Point2f& point, const cv::Point2f& center, float angleRad) {
        float cos_a = cos(angleRad);
        float sin_a = sin(angleRad);
        float dx = point.x - center.x;
        float dy = point.y - center.y;
        return cv::Point2f(
            center.x + dx * cos_a - dy * sin_a,
            center.y + dx * sin_a + dy * cos_a
        );
    }

    /**
     * Rotate image around its center
     */
    Mat rotateImage(const Mat& img, float angleDeg) {
        if (abs(angleDeg) < 0.1f) return img;  // No rotation needed

        float angleRad = angleDeg * CV_PI / 180.0f;
        Point2f center(img.cols / 2.0f, img.rows / 2.0f);
        Mat rot = getRotationMatrix2D(center, angleDeg, 1.0);
        
        // Calculate new bounding dimensions to prevent clipping
        float cos_a = abs(cos(angleRad));
        float sin_a = abs(sin(angleRad));
        int newW = static_cast<int>(img.rows * sin_a + img.cols * cos_a);
        int newH = static_cast<int>(img.rows * cos_a + img.cols * sin_a);

        // Adjust rotation matrix for new size
        rot.at<double>(0, 2) += (newW / 2.0 - center.x);
        rot.at<double>(1, 2) += (newH / 2.0 - center.y);

        Mat rotated;
        warpAffine(img, rotated, rot, Size(newW, newH), INTER_LINEAR, BORDER_REPLICATE);
        return rotated;
    }

    AnimationFrame interpolateFrame(
        const cv::Point2f& startPos,
        const cv::Point2f& endPos,
        float startRot,
        float endRot,
        float t)
    {
        // Apply easing to time parameter
        float easedT = easeInOutCubic(t);

        AnimationFrame frame;

        // Linear interpolation of position
        frame.position.x = startPos.x * (1.0f - easedT) + endPos.x * easedT;
        frame.position.y = startPos.y * (1.0f - easedT) + endPos.y * easedT;

        // Spherical interpolation of rotation (shorter path)
        float rotDiff = endRot - startRot;
        // Normalize angle difference to [-180, 180]
        while (rotDiff > 180.0f) rotDiff -= 360.0f;
        while (rotDiff < -180.0f) rotDiff += 360.0f;
        frame.rotation = startRot + rotDiff * easedT;

        frame.scale = 1.0f;

        return frame;
    }

    void animatePuzzleAssembly(
        const std::vector<PieceFeature>& features,
        const PuzzleLayout& layout,
        int canvasW,
        int canvasH,
        const AnimationConfig& config)
    {
        cout << "Starting puzzle assembly animation..." << endl;
        cout << "Total frames: " << config.totalFrames << ", FPS: " << config.fps << endl;

        // Compute initial positions (scrambled/rotated state)
        vector<Point2f> startPositions(features.size());
        vector<float> startRotations(features.size(), 0.0f);

        // For simplicity, start pieces at random positions around canvas edges
        srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < features.size(); i++) {
            int side = i % 4;
            switch (side) {
                case 0:  // Top edge
                    startPositions[i] = Point2f(rand() % canvasW, -100);
                    break;
                case 1:  // Right edge
                    startPositions[i] = Point2f(canvasW + 100, rand() % canvasH);
                    break;
                case 2:  // Bottom edge
                    startPositions[i] = Point2f(rand() % canvasW, canvasH + 100);
                    break;
                case 3:  // Left edge
                    startPositions[i] = Point2f(-100, rand() % canvasH);
                    break;
            }
            startRotations[i] = (rand() % 360);  // Random initial rotation
        }

        // Compute final positions and rotations from layout
        float minX = 1e9f, minY = 1e9f;
        for (const auto& entry : layout.positions) {
            minX = min(minX, entry.second.position.x);
            minY = min(minY, entry.second.position.y);
        }

        vector<Point2f> endPositions(features.size());
        vector<float> endRotations(features.size(), 0.0f);

        for (const auto& entry : layout.positions) {
            int pieceId = entry.first;
            const PiecePosition& pos = entry.second;
            endPositions[pieceId] = Point2f(
                pos.position.x - minX + 50,
                pos.position.y - minY + 50
            );
            endRotations[pieceId] = pos.rotation;
        }

        // Create output directory if saving frames
        if (config.saveFrames) {
            fs::create_directories(config.outputDir);
            cout << "Saving frames to: " << config.outputDir << endl;
        }

        // Animation loop
        for (int frame = 0; frame < config.totalFrames; frame++) {
            float t = static_cast<float>(frame) / (config.totalFrames - 1);

            Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));

            // Draw each piece
            for (int i = 0; i < features.size(); i++) {
                AnimationFrame animFrame = interpolateFrame(
                    startPositions[i],
                    endPositions[i],
                    startRotations[i],
                    endRotations[i],
                    t
                );

                const Mat& piece = features[i].img;
                Mat rotatedPiece = rotateImage(piece, animFrame.rotation);

                int screenX = static_cast<int>(animFrame.position.x);
                int screenY = static_cast<int>(animFrame.position.y);

                // Center the rotated piece at the position
                int startX = screenX - rotatedPiece.cols / 2;
                int startY = screenY - rotatedPiece.rows / 2;

                // Check bounds and blend if necessary
                int srcX = 0, srcY = 0;
                int dstX = startX, dstY = startY;
                int width = rotatedPiece.cols, height = rotatedPiece.rows;

                // Clip to canvas bounds
                if (dstX < 0) {
                    srcX = -dstX;
                    dstX = 0;
                }
                if (dstY < 0) {
                    srcY = -dstY;
                    dstY = 0;
                }
                if (dstX + width > canvasW) {
                    width = canvasW - dstX;
                }
                if (dstY + height > canvasH) {
                    height = canvasH - dstY;
                }

                // Draw with alpha blending for smoother appearance
                if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0) {
                    try {
                        Rect srcRect(srcX, srcY, width, height);
                        Rect dstRect(dstX, dstY, width, height);

                        Mat roi = rotatedPiece(srcRect);
                        Mat canvasRoi = canvas(dstRect);

                        // Blend with slight transparency during animation
                        double alpha = 0.8 + 0.2 * t;  // Fade in from 80% to 100%
                        addWeighted(roi, alpha, canvasRoi, 1.0 - alpha, 0, canvasRoi);
                    } catch (...) {
                        // Skip if bounds error
                    }
                }

                // Draw piece ID
                putText(canvas, to_string(i), Point(screenX + 5, screenY + 5),
                        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
            }

            // Add progress text
            putText(canvas, "Assembling Puzzle - Frame " + to_string(frame) + "/" + to_string(config.totalFrames - 1),
                    Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            putText(canvas, "Progress: " + to_string(static_cast<int>(t * 100)) + "%",
                    Point(20, canvasH - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);

            // Display frame
            if (config.showWindow) {
                imshow("Puzzle Animation", canvas);
                int delay = max(1, 1000 / config.fps);  // Milliseconds per frame
                int key = waitKey(delay);
                if (key == 27) break;  // ESC to exit
            }

            // Save frame
            if (config.saveFrames) {
                string filename = config.outputDir + "frame_" + 
                    string(5 - to_string(frame).length(), '0') + to_string(frame) + ".png";
                imwrite(filename, canvas);
            }
        }

        cout << "Animation complete!" << endl;

        if (config.showWindow) {
            destroyWindow("Puzzle Animation");
        }
    }

}
