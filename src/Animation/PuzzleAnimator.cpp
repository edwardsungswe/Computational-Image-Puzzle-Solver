#include "PuzzleAnimator.h"
#include "../Matching/PieceMatcher.h"
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

namespace PuzzleAnimator {

    float easeInOutCubic(float t) {
        return t < 0.5f ? 4.0f * t * t * t : 1.0f - pow(-2.0f * t + 2.0f, 3) / 2.0f;
    }

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

    Mat rotateImage(const Mat& img, float angleDeg) {
        if (abs(angleDeg) < 0.1f) return img;

        float angleRad = angleDeg * CV_PI / 180.0f;
        Point2f center(img.cols / 2.0f, img.rows / 2.0f);
        Mat rot = getRotationMatrix2D(center, angleDeg, 1.0);

        float cos_a = abs(cos(angleRad));
        float sin_a = abs(sin(angleRad));
        int newW = static_cast<int>(img.rows * sin_a + img.cols * cos_a);
        int newH = static_cast<int>(img.rows * cos_a + img.cols * sin_a);

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
        float easedT = easeInOutCubic(t);
        AnimationFrame frame;

        frame.position.x = startPos.x * (1.0f - easedT) + endPos.x * easedT;
        frame.position.y = startPos.y * (1.0f - easedT) + endPos.y * easedT;

        float rotDiff = endRot - startRot;
        while (rotDiff > 180.0f) rotDiff -= 360.0f;
        while (rotDiff < -180.0f) rotDiff += 360.0f;
        frame.rotation = startRot + rotDiff * easedT;
        frame.scale = 1.0f;

        return frame;
    }

    static void generateRandomStartPositions(
        vector<Point2f>& positions,
        vector<float>& rotations,
        int count,
        int canvasW,
        int canvasH)
    {
        positions.resize(count);
        rotations.resize(count);
        srand(42);

        for (int i = 0; i < count; i++) {
            int side = i % 4;
            switch (side) {
                case 0: positions[i] = Point2f(rand() % canvasW, -100); break;
                case 1: positions[i] = Point2f(canvasW + 100, rand() % canvasH); break;
                case 2: positions[i] = Point2f(rand() % canvasW, canvasH + 100); break;
                case 3: positions[i] = Point2f(-100, rand() % canvasH); break;
            }
            rotations[i] = static_cast<float>(rand() % 360);
        }
    }

    static void computeEndPositions(
        const PuzzleLayout& layout,
        const vector<PieceFeature>& features,
        vector<Point2f>& endPositions,
        vector<float>& endRotations)
    {
        float minX = 1e9f, minY = 1e9f;
        for (const auto& entry : layout.positions) {
            minX = min(minX, entry.second.position.x);
            minY = min(minY, entry.second.position.y);
        }

        endPositions.resize(features.size());
        endRotations.resize(features.size(), 0.0f);

        for (const auto& entry : layout.positions) {
            int pieceId = entry.first;
            const PiecePosition& pos = entry.second;
            endPositions[pieceId] = Point2f(pos.position.x - minX + 50, pos.position.y - minY + 50);
            endRotations[pieceId] = pos.rotation;
        }
    }

    static void renderFrame(
        Mat& canvas,
        const vector<PieceFeature>& features,
        const vector<Point2f>& startPositions,
        const vector<Point2f>& endPositions,
        const vector<float>& startRotations,
        const vector<float>& endRotations,
        float t,
        int frame,
        int totalFrames)
    {
        canvas.setTo(Scalar(0, 0, 0));

        for (size_t i = 0; i < features.size(); i++) {
            AnimationFrame animFrame = interpolateFrame(
                startPositions[i], endPositions[i],
                startRotations[i], endRotations[i], t
            );

            const Mat& piece = features[i].img;
            // The piece image is already normalized (rotated to 0 degrees)
            // So we need to apply the interpolated rotation directly
            Mat rotatedPiece = rotateImage(piece, animFrame.rotation);

            int screenX = static_cast<int>(animFrame.position.x);
            int screenY = static_cast<int>(animFrame.position.y);
            int startX = screenX - rotatedPiece.cols / 2;
            int startY = screenY - rotatedPiece.rows / 2;

            int srcX = 0, srcY = 0;
            int dstX = startX, dstY = startY;
            int width = rotatedPiece.cols, height = rotatedPiece.rows;

            if (dstX < 0) { srcX = -dstX; dstX = 0; }
            if (dstY < 0) { srcY = -dstY; dstY = 0; }
            if (dstX + width > canvas.cols) width = canvas.cols - dstX;
            if (dstY + height > canvas.rows) height = canvas.rows - dstY;

            if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0) {
                try {
                    Rect srcRect(srcX, srcY, width, height);
                    Rect dstRect(dstX, dstY, width, height);
                    Mat roi = rotatedPiece(srcRect);
                    Mat canvasRoi = canvas(dstRect);
                    double alpha = 0.8 + 0.2 * t;
                    addWeighted(roi, alpha, canvasRoi, 1.0 - alpha, 0, canvasRoi);
                } catch (...) {}
            }

            putText(canvas, to_string(i), Point(screenX + 5, screenY + 5),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
        }

        putText(canvas, "Assembling Puzzle - Frame " + to_string(frame) + "/" + to_string(totalFrames - 1),
                Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        putText(canvas, "Progress: " + to_string(static_cast<int>(t * 100)) + "%",
                Point(20, canvas.rows - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);
    }

    // Render frame with original piece images (for showing original rotation state)
    static void renderFrameWithOriginalPieces(
        Mat& canvas,
        const vector<PieceInfo>& pieceInfos,
        const vector<Point2f>& startPositions,
        const vector<Point2f>& endPositions,
        const vector<float>& startRotations,
        const vector<float>& endRotations,
        float t,
        int frame,
        int totalFrames)
    {
        canvas.setTo(Scalar(0, 0, 0));

        for (size_t i = 0; i < pieceInfos.size(); i++) {
            AnimationFrame animFrame = interpolateFrame(
                startPositions[i], endPositions[i],
                startRotations[i], endRotations[i], t
            );

            // pieceInfos[i].img is normalized (rotated to 0 degrees)
            // originalRotation is the rotation in the original image
            // To show original state: rotate normalized image by +originalRotation
            // To show final state: rotate normalized image by endRotations[i]
            // Animation: interpolate from originalRotation to endRotations[i]
            const Mat& normalizedPiece = pieceInfos[i].img;
            float originalRot = pieceInfos[i].originalRotation;
            
            // The interpolated rotation animFrame.rotation goes from originalRot to endRotations[i]
            // So we directly apply animFrame.rotation to the normalized piece
            // This will show original state at t=0 (rotation = originalRot)
            // and final state at t=1 (rotation = endRotations[i])
            Mat rotatedPiece = rotateImage(normalizedPiece, animFrame.rotation);

            int screenX = static_cast<int>(animFrame.position.x);
            int screenY = static_cast<int>(animFrame.position.y);
            int startX = screenX - rotatedPiece.cols / 2;
            int startY = screenY - rotatedPiece.rows / 2;

            int srcX = 0, srcY = 0;
            int dstX = startX, dstY = startY;
            int width = rotatedPiece.cols, height = rotatedPiece.rows;

            if (dstX < 0) { srcX = -dstX; dstX = 0; }
            if (dstY < 0) { srcY = -dstY; dstY = 0; }
            if (dstX + width > canvas.cols) width = canvas.cols - dstX;
            if (dstY + height > canvas.rows) height = canvas.rows - dstY;

            if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0) {
                try {
                    Rect srcRect(srcX, srcY, width, height);
                    Rect dstRect(dstX, dstY, width, height);
                    Mat roi = rotatedPiece(srcRect);
                    Mat canvasRoi = canvas(dstRect);
                    double alpha = 0.8 + 0.2 * t;
                    addWeighted(roi, alpha, canvasRoi, 1.0 - alpha, 0, canvasRoi);
                } catch (...) {}
            }

            putText(canvas, to_string(i), Point(screenX + 5, screenY + 5),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
        }

        putText(canvas, "Assembling Puzzle - Frame " + to_string(frame) + "/" + to_string(totalFrames - 1),
                Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        putText(canvas, "Progress: " + to_string(static_cast<int>(t * 100)) + "%",
                Point(20, canvas.rows - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);
    }

    void animatePuzzleAssembly(
        const std::vector<PieceFeature>& features,
        const PuzzleLayout& layout,
        int canvasW,
        int canvasH,
        const std::vector<cv::Point2f>& initialPositions,
        const std::vector<float>& initialRotations,
        const AnimationConfig& config)
    {
        vector<Point2f> startPositions = initialPositions;
        vector<float> startRotations = initialRotations;

        if (startPositions.size() != features.size()) {
            generateRandomStartPositions(startPositions, startRotations, features.size(), canvasW, canvasH);
        }

        vector<Point2f> endPositions;
        vector<float> endRotations;
        computeEndPositions(layout, features, endPositions, endRotations);

        if (config.saveFrames) {
            fs::create_directories(config.outputDir);
        }

        for (int frame = 0; frame < config.totalFrames; frame++) {
            float t = static_cast<float>(frame) / (config.totalFrames - 1);
            Mat canvas(canvasH, canvasW, CV_8UC3);

            renderFrame(canvas, features, startPositions, endPositions,
                        startRotations, endRotations, t, frame, config.totalFrames);

            if (config.showWindow) {
                imshow("Puzzle Animation", canvas);
                int delay = max(1, 1000 / config.fps);
                if (waitKey(delay) == 27) break;
            }

            if (config.saveFrames) {
                string filename = config.outputDir + "frame_" +
                    string(5 - to_string(frame).length(), '0') + to_string(frame) + ".png";
                imwrite(filename, canvas);
            }
        }

        if (config.showWindow) {
            destroyWindow("Puzzle Animation");
        }
    }

    void animatePuzzleAssembly(
        const std::vector<PieceFeature>& features,
        const PuzzleLayout& layout,
        int canvasW,
        int canvasH,
        const AnimationConfig& config)
    {
        vector<Point2f> startPositions;
        vector<float> startRotations;
        generateRandomStartPositions(startPositions, startRotations, features.size(), canvasW, canvasH);

        vector<Point2f> endPositions;
        vector<float> endRotations;
        computeEndPositions(layout, features, endPositions, endRotations);

        if (config.saveFrames) {
            fs::create_directories(config.outputDir);
        }

        for (int frame = 0; frame < config.totalFrames; frame++) {
            float t = static_cast<float>(frame) / (config.totalFrames - 1);
            Mat canvas(canvasH, canvasW, CV_8UC3);

            renderFrame(canvas, features, startPositions, endPositions,
                        startRotations, endRotations, t, frame, config.totalFrames);

            if (config.showWindow) {
                imshow("Puzzle Animation", canvas);
                int delay = max(1, 1000 / config.fps);
                if (waitKey(delay) == 27) break;
            }

            if (config.saveFrames) {
                string filename = config.outputDir + "frame_" +
                    string(5 - to_string(frame).length(), '0') + to_string(frame) + ".png";
                imwrite(filename, canvas);
            }
        }

        if (config.showWindow) {
            destroyWindow("Puzzle Animation");
        }
    }

    void animateSolvingProcess(
        const std::vector<PieceFeature>& features,
        const std::vector<SolvingStep>& steps,
        int canvasW,
        int canvasH,
        const AnimationConfig& config)
    {
        if (steps.empty()) return;

        int rows = steps.back().grid.size();
        if (rows == 0) return;
        int cols = steps.back().grid[0].size();

        // Calculate layout positions for each step
        vector<vector<Point2f>> stepPositions;
        vector<vector<float>> stepRotations;

        for (const auto& step : steps) {
            vector<Point2f> positions(features.size(), Point2f(-1000, -1000));
            vector<float> rotations(features.size(), 0.0f);

            // Calculate positions based on grid
            vector<int> allWidths, allHeights;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    int piece = step.grid[r][c];
                    if (piece >= 0 && piece < static_cast<int>(features.size())) {
                        Size sz = features[piece].img.size();
                        allWidths.push_back(sz.width);
                        allHeights.push_back(sz.height);
                    }
                }
            }

            if (allWidths.empty()) {
                stepPositions.push_back(positions);
                stepRotations.push_back(rotations);
                continue;
            }

            sort(allWidths.begin(), allWidths.end());
            sort(allHeights.begin(), allHeights.end());
            float medianW = allWidths[allWidths.size()/2];
            float medianH = allHeights[allHeights.size()/2];

            vector<float> yOff(rows + 1, 0);
            vector<float> xOff(cols + 1, 0);

            for (int r = 0; r < rows; r++) {
                float maxHeight = 0;
                for (int c = 0; c < cols; c++) {
                    int piece = step.grid[r][c];
                    if (piece >= 0 && piece < static_cast<int>(features.size())) {
                        Size sz = features[piece].img.size();
                        maxHeight = max(maxHeight, (float)sz.height);
                    }
                }
                yOff[r + 1] = yOff[r] + maxHeight - 1;
            }

            for (int c = 0; c < cols; c++) {
                float maxWidth = 0;
                for (int r = 0; r < rows; r++) {
                    int piece = step.grid[r][c];
                    if (piece >= 0 && piece < static_cast<int>(features.size())) {
                        Size sz = features[piece].img.size();
                        maxWidth = max(maxWidth, (float)sz.width);
                    }
                }
                xOff[c + 1] = xOff[c] + maxWidth - 1;
            }

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    int piece = step.grid[r][c];
                    if (piece >= 0 && piece < static_cast<int>(features.size())) {
                        float centerX = (xOff[c] + xOff[c + 1]) / 2.0f;
                        float centerY = (yOff[r] + yOff[r + 1]) / 2.0f;
                        positions[piece] = Point2f(centerX, centerY);
                        auto rotIt = step.rotations.find(piece);
                        if (rotIt != step.rotations.end()) {
                            rotations[piece] = rotIt->second * 90.0f;
                        }
                    }
                }
            }

            stepPositions.push_back(positions);
            stepRotations.push_back(rotations);
        }

        if (config.saveFrames) {
            fs::create_directories(config.outputDir);
        }

        // Animate between steps
        int framesPerStep = config.totalFrames / max(1, static_cast<int>(steps.size()) - 1);
        int frameCount = 0;

        for (size_t stepIdx = 0; stepIdx < steps.size() - 1; stepIdx++) {
            const auto& startPos = stepPositions[stepIdx];
            const auto& startRot = stepRotations[stepIdx];
            const auto& endPos = stepPositions[stepIdx + 1];
            const auto& endRot = stepRotations[stepIdx + 1];

            int framesThisStep = (stepIdx == steps.size() - 2) ? 
                (config.totalFrames - frameCount) : framesPerStep;

            for (int f = 0; f < framesThisStep; f++) {
                float t = static_cast<float>(f) / max(1, framesThisStep - 1);
                Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));

                // Draw all pieces that are placed (not at -1000, -1000)
                for (size_t i = 0; i < features.size(); i++) {
                    Point2f startPos_i = startPos[i];
                    Point2f endPos_i = endPos[i];
                    
                    // Only draw if piece is placed (not at default invalid position)
                    if (startPos_i.x > -500 || endPos_i.x > -500) {
                        AnimationFrame animFrame = interpolateFrame(
                            startPos_i, endPos_i,
                            startRot[i], endRot[i], t
                        );

                        // Skip if still at invalid position
                        if (animFrame.position.x < -500) continue;

                        const Mat& piece = features[i].img;
                        Mat rotatedPiece = rotateImage(piece, animFrame.rotation);

                        int screenX = static_cast<int>(animFrame.position.x);
                        int screenY = static_cast<int>(animFrame.position.y);
                        int startX = screenX - rotatedPiece.cols / 2;
                        int startY = screenY - rotatedPiece.rows / 2;

                        int srcX = 0, srcY = 0;
                        int dstX = startX, dstY = startY;
                        int width = rotatedPiece.cols, height = rotatedPiece.rows;

                        if (dstX < 0) { srcX = -dstX; width += dstX; dstX = 0; }
                        if (dstY < 0) { srcY = -dstY; height += dstY; dstY = 0; }
                        if (dstX + width > canvas.cols) width = canvas.cols - dstX;
                        if (dstY + height > canvas.rows) height = canvas.rows - dstY;

                        if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0 && 
                            srcX < rotatedPiece.cols && srcY < rotatedPiece.rows) {
                            try {
                                Rect srcRect(srcX, srcY, width, height);
                                Rect dstRect(dstX, dstY, width, height);
                                rotatedPiece(srcRect).copyTo(canvas(dstRect));
                            } catch (...) {}
                        }
                    }
                }

                // Draw step info
                string stepText = "Step " + to_string(steps[stepIdx].stepNumber + 1) + 
                                 "/" + to_string(steps.size()) + 
                                 " - Placed: " + to_string(steps[stepIdx].filledCount);
                putText(canvas, stepText, Point(20, 40), 
                       FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

                if (config.showWindow) {
                    imshow("Solving Process", canvas);
                    int delay = max(1, 1000 / config.fps);
                    if (waitKey(delay) == 27) return;
                }

                if (config.saveFrames) {
                    string filename = config.outputDir + "solve_frame_" +
                        string(5 - to_string(frameCount).length(), '0') + to_string(frameCount) + ".png";
                    imwrite(filename, canvas);
                }

                frameCount++;
            }
        }

        if (config.showWindow) {
            destroyWindow("Solving Process");
        }
    }

    void animatePuzzleAssembly(
        const std::vector<PieceFeature>& features,
        const PuzzleLayout& layout,
        const std::vector<PieceInfo>& pieceInfos,
        int canvasW,
        int canvasH,
        const AnimationConfig& config)
    {
        if (pieceInfos.size() != features.size()) {
            // Fallback to random positions if sizes don't match
            animatePuzzleAssembly(features, layout, canvasW, canvasH, config);
            return;
        }

        vector<Point2f> startPositions;
        vector<float> startRotations;
        startPositions.reserve(features.size());
        startRotations.reserve(features.size());

        // Use original positions from pieceInfos
        for (size_t i = 0; i < features.size(); i++) {
            startPositions.push_back(pieceInfos[i].center);
            startRotations.push_back(pieceInfos[i].originalRotation);
        }

        vector<Point2f> endPositions;
        vector<float> endRotations;
        computeEndPositions(layout, features, endPositions, endRotations);

        if (config.saveFrames) {
            fs::create_directories(config.outputDir);
        }

        for (int frame = 0; frame < config.totalFrames; frame++) {
            float t = static_cast<float>(frame) / (config.totalFrames - 1);
            Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));

            // Render each piece with proper rotation
            for (size_t i = 0; i < pieceInfos.size(); i++) {
                AnimationFrame animFrame = interpolateFrame(
                    startPositions[i], endPositions[i],
                    startRotations[i], endRotations[i], t
                );

                // pieceInfos[i].img is normalized (rotated to 0 degrees)
                // At t=0: animFrame.rotation = startRotations[i] = originalRotation
                // So we rotate normalized piece by originalRotation to show original state
                // At t=1: animFrame.rotation = endRotations[i]
                // So we rotate normalized piece by endRotations[i] to show final state
                const Mat& normalizedPiece = pieceInfos[i].img;
                Mat rotatedPiece = rotateImage(normalizedPiece, animFrame.rotation);

                int screenX = static_cast<int>(animFrame.position.x);
                int screenY = static_cast<int>(animFrame.position.y);
                int startX = screenX - rotatedPiece.cols / 2;
                int startY = screenY - rotatedPiece.rows / 2;

                int srcX = 0, srcY = 0;
                int dstX = startX, dstY = startY;
                int width = rotatedPiece.cols, height = rotatedPiece.rows;

                if (dstX < 0) { srcX = -dstX; dstX = 0; width += dstX; }
                if (dstY < 0) { srcY = -dstY; dstY = 0; height += dstY; }
                if (dstX + width > canvas.cols) width = canvas.cols - dstX;
                if (dstY + height > canvas.rows) height = canvas.rows - dstY;

                if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0 &&
                    srcX < rotatedPiece.cols && srcY < rotatedPiece.rows) {
                    try {
                        Rect srcRect(srcX, srcY, width, height);
                        Rect dstRect(dstX, dstY, width, height);
                        Mat roi = rotatedPiece(srcRect);
                        Mat canvasRoi = canvas(dstRect);
                        double alpha = 0.8 + 0.2 * t;
                        addWeighted(roi, alpha, canvasRoi, 1.0 - alpha, 0, canvasRoi);
                    } catch (...) {}
                }
            }

            putText(canvas, "Assembling Puzzle - Frame " + to_string(frame) + "/" + to_string(config.totalFrames - 1),
                    Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            putText(canvas, "Progress: " + to_string(static_cast<int>(t * 100)) + "%",
                    Point(20, canvas.rows - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);

            if (config.showWindow) {
                imshow("Puzzle Assembly Animation", canvas);
                int delay = max(1, 1000 / config.fps);
                if (waitKey(delay) == 27) break;
            }

            if (config.saveFrames) {
                string filename = config.outputDir + "assembly_frame_" +
                    string(5 - to_string(frame).length(), '0') + to_string(frame) + ".png";
                imwrite(filename, canvas);
            }
        }

        if (config.showWindow) {
            destroyWindow("Puzzle Assembly Animation");
        }
    }

    void showCompleteProcess(
        const cv::Mat& originalImage,
        const std::vector<PieceInfo>& pieceInfos,
        const std::vector<PieceFeature>& features,
        const std::vector<SolvingStep>& solvingSteps,
        const PuzzleLayout& layout,
        int canvasW,
        int canvasH,
        const AnimationConfig& config)
    {
        cout << "\n=== Complete Process Visualization ===" << endl;
        cout << "Press any key to continue to next step..." << endl;

        // Step 1: Show original image
        cout << "\n[Step 1/4] Showing original image..." << endl;
        Mat displayImg = originalImage.clone();
        putText(displayImg, "Step 1: Original Image", Point(20, 40),
               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        putText(displayImg, "Press any key to continue...", Point(20, displayImg.rows - 20),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        imshow("Puzzle Process", displayImg);
        waitKey(0);

        // Step 2: Show contour detection
        cout << "[Step 2/4] Showing contour detection..." << endl;
        Mat contourImg = PieceExtractor::drawContoursOnImage(originalImage);
        putText(contourImg, "Step 2: Contour Detection - Found " + to_string(pieceInfos.size()) + " pieces",
               Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        putText(contourImg, "Press any key to start extraction animation...", Point(20, contourImg.rows - 20),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        imshow("Puzzle Process", contourImg);
        waitKey(0);

        // Step 3: Animate from original image to normalized pieces in grid layout
        cout << "[Step 3/4] Animating extraction and normalization..." << endl;
        
        // Calculate grid layout for displaying pieces
        int maxPieceWidth = 0, maxPieceHeight = 0;
        for (const auto& piece : features) {
            maxPieceWidth = max(maxPieceWidth, piece.img.cols);
            maxPieceHeight = max(maxPieceHeight, piece.img.rows);
        }
        
        int colsPerRow = static_cast<int>(sqrt(pieceInfos.size())) + 1;
        int spacingX = maxPieceWidth + 20;
        int spacingY = maxPieceHeight + 20;
        
        // Calculate end positions (grid layout)
        vector<Point2f> endPositions;
        vector<float> endRotations;
        endPositions.reserve(pieceInfos.size());
        endRotations.reserve(pieceInfos.size());
        
        for (size_t i = 0; i < pieceInfos.size(); i++) {
            int row = i / colsPerRow;
            int col = i % colsPerRow;
            int offsetX = col * spacingX + 50;
            int offsetY = row * spacingY + 80;
            endPositions.push_back(Point2f(offsetX + features[i].img.cols / 2.0f, 
                                           offsetY + features[i].img.rows / 2.0f));
            endRotations.push_back(0.0f);  // Normalized pieces are at 0 rotation
        }
        
        // Start positions: original positions and rotations
        vector<Point2f> startPositions;
        vector<float> startRotations;
        startPositions.reserve(pieceInfos.size());
        startRotations.reserve(pieceInfos.size());
        
        for (size_t i = 0; i < pieceInfos.size(); i++) {
            startPositions.push_back(pieceInfos[i].center);
            // Store original rotation - this is the angle before normalization
            // The normalized image needs to be rotated by this angle to show original state
            startRotations.push_back(pieceInfos[i].originalRotation);
        }
        
        // Animate from original to normalized grid in two phases:
        // Phase 1: Rotate in place (no movement) - each piece rotates at its original position
        // Phase 2: Move to grid position (already normalized, rotation = 0)
        const int ROTATION_FRAMES = 60;  // Frames for rotation phase
        const int MOVEMENT_FRAMES = 60;  // Frames for movement phase
        const int TOTAL_FRAMES = ROTATION_FRAMES + MOVEMENT_FRAMES;
        
        for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
            Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));
            
            for (size_t i = 0; i < pieceInfos.size(); i++) {
                float currentRotation;
                Point2f currentPosition;
                
                if (frame < ROTATION_FRAMES) {
                    // Phase 1: Rotate in place (at original position)
                    float tRot = static_cast<float>(frame) / (ROTATION_FRAMES - 1);
                    currentPosition = startPositions[i];  // Stay at original position
                    // Interpolate rotation from originalRotation to 0
                    AnimationFrame rotFrame = interpolateFrame(
                        startPositions[i], startPositions[i],  // Same position
                        startRotations[i], 0.0f, tRot  // Rotate from originalRotation to 0
                    );
                    currentRotation = rotFrame.rotation;
                } else {
                    // Phase 2: Move to grid position (already normalized, rotation = 0)
                    float tMove = static_cast<float>(frame - ROTATION_FRAMES) / (MOVEMENT_FRAMES - 1);
                    currentRotation = 0.0f;  // Already normalized
                    // Interpolate position from original to grid
                    AnimationFrame moveFrame = interpolateFrame(
                        startPositions[i], endPositions[i],
                        0.0f, 0.0f, tMove  // No rotation change, just movement
                    );
                    currentPosition = moveFrame.position;
                }
                
                // pieceInfos[i].img is normalized (rotated to 0 degrees during extraction)
                // Apply the current rotation to show the piece state
                const Mat& normalizedPiece = pieceInfos[i].img;
                Mat rotatedPiece = rotateImage(normalizedPiece, currentRotation);
                
                int screenX = static_cast<int>(currentPosition.x);
                int screenY = static_cast<int>(currentPosition.y);
                int startX = screenX - rotatedPiece.cols / 2;
                int startY = screenY - rotatedPiece.rows / 2;
                
                int srcX = 0, srcY = 0;
                int dstX = startX, dstY = startY;
                int width = rotatedPiece.cols, height = rotatedPiece.rows;
                
                if (dstX < 0) { srcX = -dstX; dstX = 0; width += dstX; }
                if (dstY < 0) { srcY = -dstY; dstY = 0; height += dstY; }
                if (dstX + width > canvas.cols) width = canvas.cols - dstX;
                if (dstY + height > canvas.rows) height = canvas.rows - dstY;
                
                if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0 &&
                    srcX < rotatedPiece.cols && srcY < rotatedPiece.rows) {
                    try {
                        Rect srcRect(srcX, srcY, width, height);
                        Rect dstRect(dstX, dstY, width, height);
                        rotatedPiece(srcRect).copyTo(canvas(dstRect));
                        
                        // Draw piece number
                        string pieceNum = to_string(i);
                        putText(canvas, pieceNum, 
                               Point(screenX + 5, screenY + 5),
                               FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
                    } catch (...) {}
                }
            }
            
            string phaseText;
            if (frame < ROTATION_FRAMES) {
                phaseText = "Phase 1: Rotating pieces in place";
            } else {
                phaseText = "Phase 2: Arranging pieces in grid";
            }
            putText(canvas, "Step 3: " + phaseText + " - Frame " + to_string(frame) + "/" + to_string(TOTAL_FRAMES - 1),
                    Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            float progress = static_cast<float>(frame) / (TOTAL_FRAMES - 1);
            putText(canvas, "Progress: " + to_string(static_cast<int>(progress * 100)) + "%",
                    Point(20, canvas.rows - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);
            
            imshow("Puzzle Process", canvas);
            int delay = max(1, 1000 / config.fps);
            if (waitKey(delay) == 27) break;
        }
        
        // Show final normalized state
        Mat allPiecesDisplay = Mat::zeros(originalImage.size(), CV_8UC3);
        for (size_t i = 0; i < pieceInfos.size(); i++) {
            int row = i / colsPerRow;
            int col = i % colsPerRow;
            int offsetX = col * spacingX + 50;
            int offsetY = row * spacingY + 80;
            
            const Mat& piece = features[i].img;
            Rect dstRect(offsetX, offsetY, piece.cols, piece.rows);
            Rect srcRect(0, 0, min(piece.cols, allPiecesDisplay.cols - offsetX),
                        min(piece.rows, allPiecesDisplay.rows - offsetY));
            
            if (dstRect.x >= 0 && dstRect.y >= 0 && 
                dstRect.x + srcRect.width <= allPiecesDisplay.cols &&
                dstRect.y + srcRect.height <= allPiecesDisplay.rows) {
                Mat roi = piece(srcRect);
                roi.copyTo(allPiecesDisplay(dstRect));
                
                // Draw piece number with background
                string pieceNum = to_string(i);
                int baseline = 0;
                Size textSize = getTextSize(pieceNum, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                rectangle(allPiecesDisplay, 
                         Point(offsetX, offsetY - textSize.height - 5),
                         Point(offsetX + textSize.width + 10, offsetY),
                         Scalar(0, 0, 0), -1);
                putText(allPiecesDisplay, pieceNum, 
                       Point(offsetX + 5, offsetY - 5),
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            }
        }
        
        putText(allPiecesDisplay, "Step 3: All Extracted Pieces (" + to_string(pieceInfos.size()) + " pieces)",
               Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        putText(allPiecesDisplay, "Press any key to start assembly animation...", Point(20, allPiecesDisplay.rows - 20),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        imshow("Puzzle Process", allPiecesDisplay);
        waitKey(0);

        // Step 4: Animation from normalized pieces to final puzzle result
        cout << "[Step 4/4] Animating from normalized pieces to final puzzle result..." << endl;
        cout << "This will play automatically. Press ESC to skip." << endl;
        
        // Reuse endPositions and endRotations from Step 3 as start positions for Step 4
        // (They already contain the grid layout positions)
        vector<Point2f> assemblyStartPositions = endPositions;  // Grid positions from Step 3
        vector<float> assemblyStartRotations = endRotations;    // 0 rotation from Step 3
        
        // Calculate end positions: final puzzle layout positions and rotations
        vector<Point2f> assemblyEndPositions;
        vector<float> assemblyEndRotations;
        computeEndPositions(layout, features, assemblyEndPositions, assemblyEndRotations);
        
        if (config.saveFrames) {
            fs::create_directories(config.outputDir);
        }
        
        for (int frame = 0; frame < config.totalFrames; frame++) {
            float t = static_cast<float>(frame) / (config.totalFrames - 1);
            Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));
            
            // Render each piece animating from Step 3 grid to final puzzle layout
            for (size_t i = 0; i < pieceInfos.size(); i++) {
                AnimationFrame animFrame = interpolateFrame(
                    assemblyStartPositions[i], assemblyEndPositions[i],
                    assemblyStartRotations[i], assemblyEndRotations[i], t
                );
                
                // pieceInfos[i].img is normalized (at 0 degrees)
                // At t=0: animFrame.rotation = 0 (normalized state from Step 3)
                // At t=1: animFrame.rotation = endRotations[i] (final puzzle rotation)
                const Mat& normalizedPiece = pieceInfos[i].img;
                Mat rotatedPiece = rotateImage(normalizedPiece, animFrame.rotation);
                
                int screenX = static_cast<int>(animFrame.position.x);
                int screenY = static_cast<int>(animFrame.position.y);
                int startX = screenX - rotatedPiece.cols / 2;
                int startY = screenY - rotatedPiece.rows / 2;
                
                int srcX = 0, srcY = 0;
                int dstX = startX, dstY = startY;
                int width = rotatedPiece.cols, height = rotatedPiece.rows;
                
                if (dstX < 0) { srcX = -dstX; dstX = 0; width += dstX; }
                if (dstY < 0) { srcY = -dstY; dstY = 0; height += dstY; }
                if (dstX + width > canvas.cols) width = canvas.cols - dstX;
                if (dstY + height > canvas.rows) height = canvas.rows - dstY;
                
                if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0 &&
                    srcX < rotatedPiece.cols && srcY < rotatedPiece.rows) {
                    try {
                        Rect srcRect(srcX, srcY, width, height);
                        Rect dstRect(dstX, dstY, width, height);
                        Mat roi = rotatedPiece(srcRect);
                        Mat canvasRoi = canvas(dstRect);
                        double alpha = 0.8 + 0.2 * t;
                        addWeighted(roi, alpha, canvasRoi, 1.0 - alpha, 0, canvasRoi);
                    } catch (...) {}
                }
            }
            
            putText(canvas, "Assembling Puzzle - Frame " + to_string(frame) + "/" + to_string(config.totalFrames - 1),
                    Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            putText(canvas, "Progress: " + to_string(static_cast<int>(t * 100)) + "%",
                    Point(20, canvas.rows - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);
            
            if (config.showWindow) {
                imshow("Puzzle Process", canvas);
                int delay = max(1, 1000 / config.fps);
                if (waitKey(delay) == 27) break;
            }
            
            if (config.saveFrames) {
                string filename = config.outputDir + "assembly_frame_" +
                    string(5 - to_string(frame).length(), '0') + to_string(frame) + ".png";
                imwrite(filename, canvas);
            }
        }
        
        // Show final result centered and wait for key
        cout << "\nPuzzle assembled! Showing final result..." << endl;
        
        // Calculate final puzzle bounds
        float minX = 1e9f, minY = 1e9f;
        float maxX = -1e9f, maxY = -1e9f;
        for (const auto& entry : layout.positions) {
            const PiecePosition& pos = entry.second;
            minX = min(minX, pos.position.x);
            minY = min(minY, pos.position.y);
            maxX = max(maxX, pos.position.x + pos.size.width);
            maxY = max(maxY, pos.position.y + pos.size.height);
        }
        
        float layoutWidth = maxX - minX;
        float layoutHeight = maxY - minY;
        float centerX = (canvasW - layoutWidth) / 2.0f;
        float centerY = (canvasH - layoutHeight) / 2.0f;
        
        // Render final puzzle centered
        Mat finalCanvas(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));
        for (const auto& entry : layout.positions) {
            int pieceId = entry.first;
            const PiecePosition& pos = entry.second;
            const Mat& piece = features[pieceId].img;
            Mat rotatedPiece = rotateImage(piece, pos.rotation);
            
            int screenX = static_cast<int>(pos.position.x - minX + centerX);
            int screenY = static_cast<int>(pos.position.y - minY + centerY);
            int startX = screenX - rotatedPiece.cols / 2;
            int startY = screenY - rotatedPiece.rows / 2;
            
            int srcX = 0, srcY = 0;
            int dstX = startX, dstY = startY;
            int width = rotatedPiece.cols, height = rotatedPiece.rows;
            
            if (dstX < 0) { srcX = -dstX; dstX = 0; width += dstX; }
            if (dstY < 0) { srcY = -dstY; dstY = 0; height += dstY; }
            if (dstX + width > finalCanvas.cols) width = finalCanvas.cols - dstX;
            if (dstY + height > finalCanvas.rows) height = finalCanvas.rows - dstY;
            
            if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0 &&
                srcX < rotatedPiece.cols && srcY < rotatedPiece.rows) {
                try {
                    Rect srcRect(srcX, srcY, width, height);
                    Rect dstRect(dstX, dstY, width, height);
                    rotatedPiece(srcRect).copyTo(finalCanvas(dstRect));
                } catch (...) {}
            }
        }
        
        putText(finalCanvas, "Puzzle Complete!", Point(20, 40),
               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        putText(finalCanvas, "Press any key to exit...", Point(20, finalCanvas.rows - 20),
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        
        imshow("Puzzle Process", finalCanvas);
        waitKey(0);
        
        cout << "\n=== Process Visualization Complete ===" << endl;
    }

}