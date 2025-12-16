#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Features/FeatureExtractor.h"
#include "Matching/PieceMatcher.h"
#include "ImageIO/ImageLoader.h"
#include "Pieces/PieceExtractor.h"
#include "Animation/PuzzleAnimator.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <base_path>" << endl;
        return -1;
    }

    string basePath = argv[1];
    string pathRGB = basePath + ".rgb";
    string pathPNG = basePath + ".png";

    Mat img = imread(pathPNG, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error loading image: " << pathPNG << endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    unsigned char* buffer = ImageLoader::loadRawRGB(pathRGB, width, height);
    if (!buffer) return -1;

    Mat rgbImage(height, width, CV_8UC3, buffer);
    Mat bgrImage;
    cvtColor(rgbImage, bgrImage, COLOR_RGB2BGR);

    string outputDir = "./extracted_pieces";
    
    vector<PieceInfo> pieceInfos = PieceExtractor::extractPiecesWithInfo(bgrImage, outputDir);
    
    vector<PieceFeature> features;
    for (const auto& info : pieceInfos) {
        features.push_back(FeatureExtractor::extract(info.img));
    }

    vector<SolvingStep> solvingSteps;
    PuzzleLayout layout = PieceMatcher::solveWithSteps(features, width, height, solvingSteps);

    if (layout.positions.empty()) {
        cerr << "Solver failed" << endl;
        free(buffer);
        return -1;
    }

    float minX = numeric_limits<float>::max();
    float minY = numeric_limits<float>::max();
    float maxX = -numeric_limits<float>::max();
    float maxY = -numeric_limits<float>::max();

    for (const auto& entry : layout.positions) {
        const PiecePosition& pos = entry.second;
        minX = min(minX, pos.position.x);
        minY = min(minY, pos.position.y);
        maxX = max(maxX, pos.position.x + pos.size.width);
        maxY = max(maxY, pos.position.y + pos.size.height);
    }

    float layoutWidth = maxX - minX;
    float layoutHeight = maxY - minY;
    float centerX = (width - layoutWidth) / 2.0f;
    float centerY = (height - layoutHeight) / 2.0f;

    Mat finalPuzzle(height, width, CV_8UC3, Scalar(0, 0, 0));
    Mat finalPuzzleWeight(height, width, CV_32FC1, Scalar(0.0f));  // Weight accumulator for blending

    // Check if any pieces are rotated (for edge blur)
    bool hasRotation = false;
    for (const auto& entry : layout.positions) {
        if (entry.second.rotation != 0.0f) {
            hasRotation = true;
            break;
        }
    }

    // Apply edge blur to all pieces if there are rotations, to eliminate gaps
    const int EDGE_BLUR_SIZE = hasRotation ? 6 : 0;  // Increased blur size for better blending

    for (const auto& entry : layout.positions) {
        int pieceId = entry.first;
        const PiecePosition& pos = entry.second;
        const Mat& piece = features[pieceId].img;
        Mat pieceToDraw = PieceMatcher::rotatePiece(piece, pos.rotation);

        int screenX = static_cast<int>(pos.position.x - minX + centerX);
        int screenY = static_cast<int>(pos.position.y - minY + centerY);

        int pieceW = pieceToDraw.cols;
        int pieceH = pieceToDraw.rows;
        int copyW = min(pieceW, width - screenX);
        int copyH = min(pieceH, height - screenY);

        if (screenX >= 0 && screenY >= 0 && copyW > 0 && copyH > 0) {
            Rect srcROI(0, 0, copyW, copyH);
            Rect dstROI(screenX, screenY, copyW, copyH);
            
            Mat pieceROI = pieceToDraw(srcROI).clone();
            Mat canvasROI = finalPuzzle(dstROI);
            Mat weightROI = finalPuzzleWeight(dstROI);

            if (EDGE_BLUR_SIZE > 0) {
                // Create edge-blurred mask for all pieces when rotations are present
                Mat mask(pieceROI.size(), CV_32FC1, Scalar(1.0f));
                
                // Create edge mask (fade out at edges with smoother transition)
                int blurWidth = EDGE_BLUR_SIZE;
                for (int y = 0; y < mask.rows; y++) {
                    for (int x = 0; x < mask.cols; x++) {
                        float distTop = static_cast<float>(y);
                        float distBottom = static_cast<float>(mask.rows - 1 - y);
                        float distLeft = static_cast<float>(x);
                        float distRight = static_cast<float>(mask.cols - 1 - x);
                        
                        float minDist = min({distTop, distBottom, distLeft, distRight});
                        if (minDist < blurWidth) {
                            // Smooth transition: ensure minimum alpha to prevent complete transparency
                            float normalizedDist = minDist / blurWidth;
                            // Use smoother curve: start fading later, keep more opacity at edges
                            float alpha = 0.3f + 0.7f * normalizedDist * normalizedDist;  // Minimum 0.3 alpha at edges
                            mask.at<float>(y, x) = alpha;
                        }
                    }
                }
                
                // Apply stronger Gaussian blur to smooth the mask
                Mat blurredMask;
                GaussianBlur(mask, blurredMask, Size(9, 9), 2.0);
                
                // Blend using the mask with weighted average
                Mat pieceFloat, canvasFloat;
                pieceROI.convertTo(pieceFloat, CV_32FC3);
                canvasROI.convertTo(canvasFloat, CV_32FC3);
                
                for (int y = 0; y < pieceROI.rows; y++) {
                    for (int x = 0; x < pieceROI.cols; x++) {
                        float alpha = blurredMask.at<float>(y, x);
                        float oldWeight = weightROI.at<float>(y, x);
                        float newWeight = alpha;
                        float totalWeight = oldWeight + newWeight;
                        
                        Vec3f newColor = pieceFloat.at<Vec3f>(y, x);
                        
                        if (totalWeight > 0.001f) {
                            Vec3f oldColor = canvasFloat.at<Vec3f>(y, x);
                            Vec3f blendedColor = (oldColor * oldWeight + newColor * newWeight) / totalWeight;
                            canvasFloat.at<Vec3f>(y, x) = blendedColor;
                            weightROI.at<float>(y, x) = totalWeight;
                        } else {
                            canvasFloat.at<Vec3f>(y, x) = newColor;
                            weightROI.at<float>(y, x) = newWeight;
                        }
                    }
                }
                
                canvasFloat.convertTo(canvasROI, CV_8UC3);
            } else {
                // For non-rotated pieces, use simple copy (no blur needed)
                pieceROI.copyTo(canvasROI);
                weightROI.setTo(1.0f);
            }
        }
    }


    // imwrite("output.png", finalPuzzle);
    imshow("Solved Puzzle", finalPuzzle);
    waitKey(0);
    PuzzleAnimator::AnimationConfig animConfig;
    animConfig.totalFrames = 180;
    animConfig.fps = 30;
    animConfig.showWindow = true;
    animConfig.saveFrames = false;
    animConfig.outputDir = "./animation_frames/";
    
    PuzzleAnimator::showCompleteProcess(
        bgrImage,
        pieceInfos,
        features,         
        solvingSteps,      
        layout,     
        width, height,
        animConfig
    );
    
    destroyAllWindows();

    free(buffer);
    return 0;
}