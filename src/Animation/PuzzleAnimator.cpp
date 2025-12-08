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

}