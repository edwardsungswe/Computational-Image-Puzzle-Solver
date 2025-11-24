#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "FeatureExtractor.h"
#include "Matcher.h"
#include "Assembler.h"
#include "ImageIO/ImageLoader.h"
#include "Pieces/PieceExtractor.h"
#include "Animation/PuzzleAnimator.h"

using namespace std;
using namespace cv;


int main() {
    string pathRGB     = "./data_sample/starry_night_translate.rgb";
    string pathPNG  = "./data_sample/starry_night_translate.png";
    // string pathRGB     = "./data_sample/starry_night_rotate.rgb";
    // string pathPNG  = "./data_sample/starry_night_rotate.png";

    // Load helper PNG
    Mat img = imread(pathPNG, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error Loading Image: " << pathPNG << endl;
        return -1;
    }

    cout << "Image loaded, size: " << img.cols << "x" << img.rows << endl;

    int width  = img.cols;
    int height = img.rows;
    std::cout << "Detecta123123sded ";
    // Load raw RGB
    unsigned char* buffer = ImageLoader::loadRawRGB(pathRGB, width, height);
    if (!buffer) return -1;

    // Load raw .rgb
    Mat rgbImage(height, width, CV_8UC3, buffer);

    Mat bgrImage;
    cvtColor(rgbImage, bgrImage, COLOR_RGB2BGR);

    // Convert & mask
    Mat gray;
    cvtColor(rgbImage, gray, COLOR_RGB2GRAY);

    Mat mask;
    threshold(gray, mask, 10, 255, THRESH_BINARY);

    // Contours
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = bgrImage.clone();
    vector<Mat> pieces;
    vector<cv::Point2f> pieceInitialPositions;  // Track initial positions
    vector<float> pieceInitialRotations;        // Track initial rotations (all 0 for scrambled)

    for (const auto& c : contours) {
        // RotatedRect box = minAreaRect(c);
        Rect box = boundingRect(c);  // For non-rotated image

        // Draw box
        // Point2f ptsf[4];
        // box.points(ptsf);
        // vector<Point> poly;
        // for (int i = 0; i < 4; i++) {
        //     poly.emplace_back(Point(cvRound(ptsf[i].x), cvRound(ptsf[i].y)));
        // }
        // polylines(contourImage, poly, true, Scalar(0,0,255), 2);

        
        rectangle(contourImage, box, Scalar(0,0,255), 2); // For non-rotated image

        // float angle = box.angle;
        // Size2f boxSize = box.size;
        // if (box.angle < -45.0f) {
        //     angle += 90.0f;
        //     swap(boxSize.width, boxSize.height);
        // }

        // Rotate
        // Mat M = getRotationMatrix2D(box.center, angle, 1.0);
        // Mat rotated;
        // warpAffine(bgrImage, rotated, M, bgrImage.size(), INTER_LANCZOS4, BORDER_REFLECT);

        // ROI extraction
        // Rect roi;
        // roi.width  = (int)boxSize.width;
        // roi.height = (int)boxSize.height;
        // roi.x = (int)(box.center.x - roi.width  / 2);
        // roi.y = (int)(box.center.y - roi.height / 2);

        // // Apply bounds checking
        // roi.x = max(0, min(roi.x, rotated.cols - 1));
        // roi.y = max(0, min(roi.y, rotated.rows - 1));
        // if (roi.x + roi.width > rotated.cols)  roi.width  = rotated.cols - roi.x;
        // if (roi.y + roi.height > rotated.rows) roi.height = rotated.rows - roi.y;

        // if (roi.width > 0 && roi.height > 0) {
        //     // Create padded ROI to avoid edge artifacts
        //     Rect roi_padded = roi;
        //     roi_padded.x = max(0, roi.x - 1);  // Expand left
        //     roi_padded.y = max(0, roi.y - 1);  // Expand top
        //     roi_padded.width = min(rotated.cols - roi_padded.x, roi.width + 2);   // Expand right
        //     roi_padded.height = min(rotated.rows - roi_padded.y, roi.height + 2); // Expand bottom
            
        //     Mat piece_with_padding = rotated(roi_padded).clone();
            
        //     // Extract the actual piece without padding (center of the padded region)
        //     Rect inner_roi(1, 1, roi.width, roi.height);  // Offset within padded image
        //     Mat piece = piece_with_padding(inner_roi).clone();
            
        //     pieces.push_back(piece);
        // }
       

        // For non-rotated image
        int x = max(0, min(box.x, bgrImage.cols - 1));
        int y = max(0, min(box.y, bgrImage.rows - 1));
        int w = min(box.width,  bgrImage.cols - x);
        int h = min(box.height, bgrImage.rows - y);

        if (w > 0 && h > 0) {
            Mat piece = bgrImage(Rect(x, y, w, h)).clone();
            pieces.push_back(bgrImage(Rect(x, y, w, h)).clone());
            
            // Store initial position (center of the piece in the original image)
            pieceInitialPositions.push_back(cv::Point2f(x + w / 2.0f, y + h / 2.0f));
            pieceInitialRotations.push_back(0.0f);  // All pieces start with 0 rotation
            // string windowName = "Piece " + to_string(pieces.size());
            // namedWindow(windowName, WINDOW_NORMAL);
            // resizeWindow(windowName, min(400, w), min(400, h));
            // imshow(windowName, piece);
            
            // // Wait for a key press to see each piece
            // waitKey(100);
        }

    }


    cout << "Detected " << pieces.size() << " pieces." << endl;
    cout << "Detected " << pieces.size() << " pieces." << endl;

    int canvasW = img.cols;
    int canvasH = img.rows;
    Mat canvas(canvasH, canvasW, CV_8UC3);

    auto showStage = [&](const Mat& image, const string& label) {
        Mat display;
        resize(image, display, Size(canvasW, canvasH));
        display.copyTo(canvas);

        putText(canvas, label, Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);

        putText(canvas, "Press any key to continue...",
                Point(20, canvasH - 20),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200,200,200), 2);

        imshow("Puzzle Demo", canvas);
        waitKey(0);
    };

    showStage(bgrImage, "Stage 1: Original Image");

    showStage(contourImage, "Stage 2: Detected Contours");

    std::cout << "Press any key..." << std::endl;
    cv::waitKey(0);

    Mat preview(canvasH, canvasW, CV_8UC3, Scalar(0,0,0));

    int x = 10, y = 10, rowH = 0;
    int pieceID = 0;
    for (auto& piece : pieces) {
        if (x + piece.cols > canvasW - 10) {
            x = 10;
            y += rowH + 10;
            rowH = 0;
        }
        if (y + piece.rows > canvasH - 10) break;

        piece.copyTo(preview(Rect(x, y, piece.cols, piece.rows)));
        rowH = max(rowH, piece.rows);
        putText(preview, to_string(pieceID), Point(x + 5, y + 20),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
        x += piece.cols + 10;
        rowH = max(rowH, piece.rows);
        pieceID++;
    }

    // showStage(preview, "Stage 3: Corrected Pieces");
    showStage(preview, "S");


    // Testing accuracy of Matcher and building final image without animation currently
    vector<PieceFeature> features;
    for (auto& p : pieces) features.push_back(FeatureExtractor::extract(p));

    vector<Pair> allMatches = Matcher::createFilteredMatches(features, 0.8);


    PuzzleLayout layout = Matcher::buildLayout(allMatches, features);
 

    Mat finalAssembly(canvasH, canvasW, CV_8UC3, Scalar(0,0,0));

    float minX = 0, minY = 0;
    for (const auto& entry : layout.positions) {
        minX = min(minX, entry.second.position.x);
        minY = min(minY, entry.second.position.y);
    }

    for (auto& entry : layout.positions) {
        int pieceId = entry.first;
        const PiecePosition& pos = entry.second;
        const Mat& piece = features[pieceId].img;
        Mat pieceToDraw = Matcher::rotatePiece(piece, pos.rotation);

        int screenX = static_cast<int>(pos.position.x - minX + 50);
        int screenY = static_cast<int>(pos.position.y - minY + 50);

        if (screenX >= 0 && screenY >= 0 &&
            screenX + piece.cols <= img.cols &&
            screenY + piece.rows <= img.rows) {
            pieceToDraw.copyTo(finalAssembly(Rect(screenX, screenY, piece.cols, piece.rows)));
        }

        putText(finalAssembly, to_string(pieceId), Point(screenX + 5, screenY + 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
        rectangle(finalAssembly, Point(screenX, screenY),
                  Point(screenX + piece.cols, screenY + piece.rows), Scalar(0,255,0), 1);
    }
    putText(finalAssembly, "Final Matched Layout", 
            Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);
    putText(finalAssembly, "Press any key to start animation...", 
            Point(20, canvasH - 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200,200,200), 2);

    imshow("Final Layout", finalAssembly);
    waitKey(0);
    destroyWindow("Final Layout");

    // Configure and run animation
    PuzzleAnimator::AnimationConfig animConfig;
    animConfig.totalFrames = 180;     // 6 seconds at 30 FPS
    animConfig.fps = 30;
    animConfig.showWindow = true;
    animConfig.saveFrames = false;    // Set to true to save frames to disk

    PuzzleAnimator::animatePuzzleAssembly(features, layout, canvasW, canvasH, 
                                          pieceInitialPositions, pieceInitialRotations, animConfig);

    // Display final assembled puzzle and wait for keypress
    Mat finalPuzzle(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));

    for (auto& entry : layout.positions) {
        int pieceId = entry.first;
        const PiecePosition& pos = entry.second;
        const Mat& piece = features[pieceId].img;
        Mat pieceToDraw = Matcher::rotatePiece(piece, pos.rotation);

        int screenX = static_cast<int>(pos.position.x - minX + 50);
        int screenY = static_cast<int>(pos.position.y - minY + 50);

        if (screenX >= 0 && screenY >= 0 &&
            screenX + piece.cols <= canvasW &&
            screenY + piece.rows <= canvasH) {
            pieceToDraw.copyTo(finalPuzzle(Rect(screenX, screenY, piece.cols, piece.rows)));
        }
    }

    putText(finalPuzzle, "Animation Complete!", 
            Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
    putText(finalPuzzle, "Press any key to exit...", 
            Point(20, canvasH - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);

    imshow("Final Puzzle", finalPuzzle);
    waitKey(0);
    destroyWindow("Final Puzzle");

    return 0;
}
