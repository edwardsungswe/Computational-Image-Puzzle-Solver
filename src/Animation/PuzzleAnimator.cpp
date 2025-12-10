#include "PuzzleAnimator.h"
#include "../Matching/PieceMatcher.h"
#include <iostream>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

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

    // Helper function to draw rotated image directly on canvas at specified position
    // Uses proper mask to avoid black borders - only draws pixels that were part of original image
    static void drawRotatedImage(
        Mat& canvas,
        const Mat& img,
        const Point2f& centerPos,
        float angleDeg,
        double alpha = 1.0)
    {
        if (abs(angleDeg) < 0.1f) {
            // No rotation, just draw normally
            int startX = static_cast<int>(centerPos.x - img.cols / 2.0f);
            int startY = static_cast<int>(centerPos.y - img.rows / 2.0f);
            
            int srcX = 0, srcY = 0;
            int dstX = startX, dstY = startY;
            int width = img.cols, height = img.rows;
            
            if (dstX < 0) { srcX = -dstX; width += dstX; dstX = 0; }
            if (dstY < 0) { srcY = -dstY; height += dstY; dstY = 0; }
            if (dstX + width > canvas.cols) width = canvas.cols - dstX;
            if (dstY + height > canvas.rows) height = canvas.rows - dstY;
            
            if (width > 0 && height > 0 && srcX >= 0 && srcY >= 0 &&
                srcX < img.cols && srcY < img.rows) {
                try {
                    Rect srcRect(srcX, srcY, width, height);
                    Rect dstRect(dstX, dstY, width, height);
                    Mat roi = img(srcRect);
                    Mat canvasRoi = canvas(dstRect);
                    if (alpha >= 1.0) {
                        roi.copyTo(canvasRoi);
                    } else {
                        addWeighted(roi, alpha, canvasRoi, 1.0 - alpha, 0, canvasRoi);
                    }
                } catch (...) {}
            }
            return;
        }
        
        // Create rotation matrix centered at the image's own center
        Point2f imgCenter(img.cols / 2.0f, img.rows / 2.0f);
        
        // Create a mask for the original image - mark valid pixels
        // This ensures we only draw pixels that were part of the original image
        Mat originalMask;
        if (img.channels() == 4) {
            // If image has alpha channel, use it
            vector<Mat> channels;
            split(img, channels);
            originalMask = channels[3];  // Alpha channel
        } else {
            // For images without alpha channel, assume entire image is valid
            // Create a white mask (all pixels valid) to avoid filtering out black pixels in the image
            originalMask = Mat::ones(img.rows, img.cols, CV_8UC1) * 255;
        }
        
        // Create ROI on canvas where we'll draw (large enough to contain rotated image)
        float diagonal = sqrt(img.cols * img.cols + img.rows * img.rows);
        int roiSize = static_cast<int>(ceil(diagonal)) + 20;
        int roiX = max(0, static_cast<int>(centerPos.x - roiSize / 2));
        int roiY = max(0, static_cast<int>(centerPos.y - roiSize / 2));
        int roiW = min(roiSize, canvas.cols - roiX);
        int roiH = min(roiSize, canvas.rows - roiY);
        
        if (roiW <= 0 || roiH <= 0) return;
        
        // Create ROI canvas
        Rect roiRect(roiX, roiY, roiW, roiH);
        Mat roiCanvas = canvas(roiRect);
        
        // Create rotation matrix that will place rotated image center at centerPos
        Mat rotMatrix = getRotationMatrix2D(imgCenter, angleDeg, 1.0);
        
        // Adjust rotation matrix for ROI coordinate system
        Mat roiRotMatrix = rotMatrix.clone();
        roiRotMatrix.at<double>(0, 2) += centerPos.x - imgCenter.x - roiX;
        roiRotMatrix.at<double>(1, 2) += centerPos.y - imgCenter.y - roiY;
        
        // Rotate the image
        Mat rotatedImg;
        warpAffine(img, rotatedImg, roiRotMatrix, Size(roiW, roiH), 
                  INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
        
        // Rotate the mask to know which pixels are valid
        Mat rotatedMask;
        warpAffine(originalMask, rotatedMask, roiRotMatrix, Size(roiW, roiH), 
                  INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
        
        // Only draw pixels where the mask is non-zero (valid pixels from original image)
        // This prevents drawing black borders - only pixels that were in the original image are drawn
        if (alpha >= 1.0) {
            // Direct copy, only where mask is valid
            rotatedImg.copyTo(roiCanvas, rotatedMask);
        } else {
            // Blend with alpha, but only for valid pixels
            Mat blended;
            addWeighted(rotatedImg, alpha, roiCanvas, 1.0 - alpha, 0, blended);
            // Only apply blended result where mask is valid
            blended.copyTo(roiCanvas, rotatedMask);
        }
    }

    // Simplified rotateImage: keep original size to avoid scale changes
    // Note: This may crop edges when rotating, but prevents black borders
    Mat rotateImage(const Mat& img, float angleDeg) {
        if (abs(angleDeg) < 0.1f) return img.clone();

        Point2f center(img.cols / 2.0f, img.rows / 2.0f);
        Mat rot = getRotationMatrix2D(center, angleDeg, 1.0);
        
        // Keep original size - this prevents scale changes but may crop edges
        Mat rotated;
        warpAffine(img, rotated, rot, img.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
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

    // Helper function to save frame debug info to JSON
    static void saveFrameDebugInfo(
        const string& filename,
        int frame,
        float t,
        const vector<AnimationFrame>& frames)
    {
        ofstream out(filename);
        if (!out.is_open()) return;
        
        out << "{\n";
        out << "  \"frame\": " << frame << ",\n";
        out << "  \"progress\": " << t << ",\n";
        out << "  \"pieces\": [\n";
        
        for (size_t i = 0; i < frames.size(); i++) {
            out << "    {\n";
            out << "      \"id\": " << i << ",\n";
            out << "      \"position\": {\"x\": " << frames[i].position.x 
                << ", \"y\": " << frames[i].position.y << "},\n";
            out << "      \"rotation\": " << frames[i].rotation << ",\n";
            out << "      \"scale\": " << frames[i].scale << "\n";
            out << "    }";
            if (i < frames.size() - 1) out << ",";
            out << "\n";
        }
        
        out << "  ]\n";
        out << "}\n";
        out.close();
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
            // Use drawRotatedImage to draw directly on canvas, avoiding size changes
            double alpha = 0.8 + 0.2 * t;
            drawRotatedImage(canvas, piece, animFrame.position, animFrame.rotation, alpha);

            int screenX = static_cast<int>(animFrame.position.x);
            int screenY = static_cast<int>(animFrame.position.y);
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
            double alpha = 0.8 + 0.2 * t;
            drawRotatedImage(canvas, normalizedPiece, animFrame.position, animFrame.rotation, alpha);

            int screenX = static_cast<int>(animFrame.position.x);
            int screenY = static_cast<int>(animFrame.position.y);
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
                                
                                // Draw piece number
                                string pieceNum = to_string(i);
                                putText(canvas, pieceNum, 
                                       Point(screenX + 5, screenY + 5),
                                       FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
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
                double alpha = 0.8 + 0.2 * t;
                drawRotatedImage(canvas, normalizedPiece, animFrame.position, animFrame.rotation, alpha);
                
                // Draw piece number
                int screenX = static_cast<int>(animFrame.position.x);
                int screenY = static_cast<int>(animFrame.position.y);
                string pieceNum = to_string(i);
                putText(canvas, pieceNum, 
                       Point(screenX + 5, screenY + 5),
                       FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
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
            // pieceInfos[i].img is already rotated to 0 degrees during extraction
            // pieceInfos[i].originalRotation is the angle used to rotate the image during extraction
            // To show original state, we need to rotate BACK by -originalRotation
            // So start rotation should be -originalRotation, end rotation is 0
            startRotations.push_back(-pieceInfos[i].originalRotation);
        }
        
        // Animate from original to normalized: Rotate in place (no movement)
        // Each piece rotates at its original position from originalRotation to 0
        const int ROTATION_FRAMES = 60;  // Frames for rotation phase
        
        for (int frame = 0; frame < ROTATION_FRAMES; frame++) {
            Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));
            
            for (size_t i = 0; i < pieceInfos.size(); i++) {
                // Rotate in place (at original position)
                float tRot = static_cast<float>(frame) / (ROTATION_FRAMES - 1);
                Point2f currentPosition = startPositions[i];  // Stay at original position
                // Interpolate rotation from -originalRotation (shows original state) to 0 (normalized)
                AnimationFrame rotFrame = interpolateFrame(
                    startPositions[i], startPositions[i],  // Same position
                    startRotations[i], 0.0f, tRot  // Rotate from -originalRotation to 0
                );
                float currentRotation = rotFrame.rotation;
                
                // pieceInfos[i].img is normalized (rotated to 0 degrees during extraction)
                // Apply the current rotation to show the piece state
                // Use drawRotatedImage to properly handle rotation without black borders
                const Mat& normalizedPiece = pieceInfos[i].img;
                drawRotatedImage(canvas, normalizedPiece, currentPosition, currentRotation, 1.0);
                
                // Draw piece number
                int screenX = static_cast<int>(currentPosition.x);
                int screenY = static_cast<int>(currentPosition.y);
                string pieceNum = to_string(i);
                putText(canvas, pieceNum, 
                       Point(screenX + 5, screenY + 5),
                       FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
            }
            
            putText(canvas, "Step 3: Rotating pieces in place - Frame " + to_string(frame) + "/" + to_string(ROTATION_FRAMES - 1),
                    Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            float progress = static_cast<float>(frame) / (ROTATION_FRAMES - 1);
            putText(canvas, "Progress: " + to_string(static_cast<int>(progress * 100)) + "%",
                    Point(20, canvas.rows - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);
            
            imshow("Puzzle Process", canvas);
            int delay = max(1, 1000 / config.fps);
            if (waitKey(delay) == 27) break;
        }

        // Step 4: Animation from normalized pieces (at original positions) to final puzzle result
        cout << "[Step 4/4] Animating from normalized pieces to final puzzle result..." << endl;
        cout << "This will play automatically. Press ESC to skip." << endl;
        
        // Use original positions as start positions for Step 4 (pieces are now normalized/rotated to 0)
        vector<Point2f> assemblyStartPositions = startPositions;  // Original positions from Step 3
        vector<float> assemblyStartRotations;  // All pieces are now normalized (rotation = 0)
        assemblyStartRotations.resize(pieceInfos.size(), 0.0f);
        
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
            
            // Render each piece animating from original positions to final puzzle layout
            for (size_t i = 0; i < pieceInfos.size(); i++) {
                AnimationFrame animFrame = interpolateFrame(
                    assemblyStartPositions[i], assemblyEndPositions[i],
                    assemblyStartRotations[i], assemblyEndRotations[i], t
                );
                
                // pieceInfos[i].img is normalized (at 0 degrees)
                // At t=0: animFrame.rotation = 0 (normalized state from Step 3)
                // At t=1: animFrame.rotation = endRotations[i] (final puzzle rotation)
                const Mat& normalizedPiece = pieceInfos[i].img;
                double alpha = 0.8 + 0.2 * t;
                drawRotatedImage(canvas, normalizedPiece, animFrame.position, animFrame.rotation, alpha);
                
                // Draw piece number
                int screenX = static_cast<int>(animFrame.position.x);
                int screenY = static_cast<int>(animFrame.position.y);
                string pieceNum = to_string(i);
                putText(canvas, pieceNum, 
                       Point(screenX + 5, screenY + 5),
                       FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
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
                    
                    // Draw piece number
                    string pieceNum = to_string(pieceId);
                    putText(finalCanvas, pieceNum, 
                           Point(screenX + 5, screenY + 5),
                           FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
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