#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "Features/FeatureExtractor.h"
#include "Matching/PieceMatcher.h"
#include "Puzzle/Assembler.h"
#include "ImageIO/ImageLoader.h"
#include "Pieces/PieceExtractor.h"
#include "Animation/PuzzleAnimator.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    // string pathRGB     = "./data_sample/starry_night_translate.rgb"; ✅
    // string pathPNG  = "./data_sample/starry_night_translate.png";
    // string pathRGB     = "./data_sample/starry_night_rotate.rgb";
    // string pathPNG  = "./data_sample/starry_night_rotate.png";
    // string pathRGB     = "./data_sample/mona_lisa_translate.rgb";✅
    // string pathPNG  = "./data_sample/mona_lisa_translate.png";
    // string pathRGB2 = "./data_sample/mona_lisa_rotate.rgb";
    // string pathPNG2 = "./data_sample/mona_lisa_rotate.png";
    // string pathRGB     = "./data_sample/more_samples/more_samples/sample1/sample1_translate.rgb";✅
    // string pathPNG  = "./data_sample/more_samples/more_samples/sample1/sample1_translate.png";
    // string pathRGB     = "./data_sample/more_samples/more_samples/sample1/sample1_rotate.rgb";
    // string pathPNG  = "./data_sample/more_samples/more_samples/sample1/sample1_rotate.png";
    // string pathRGB     = "./data_sample/more_samples/more_samples/sample2/sample2_translate.rgb";
    // string pathPNG  = "./data_sample/more_samples/more_samples/sample2/sample2_translate.png";
    // string pathRGB     = "./data_sample/more_samples/more_samples/sample2/sample2_rotate.rgb";
    // string pathPNG  = "./data_sample/more_samples/more_samples/sample2/sample2_rotate.png";
    // string pathRGB     = "./data_sample/more_samples/more_samples/sample3/sample3_translate.rgb";✅
    // string pathPNG  = "./data_sample/more_samples/more_samples/sample3/sample3_translate.png";
    // string pathRGB     = "./data_sample/more_samples/more_samples/sample3/sample3_rotate.rgb";
    // string pathPNG  = "./data_sample/more_samples/more_samples/sample3/sample3_rotate.png";
    string basePath;
    
    if (argc >= 2) {
        basePath = argv[1];
    } else {
        cout << "Usage: " << argv[0] << " <base_path>" << endl;
        return -1;
    }
    
    string pathRGB = basePath + ".rgb";
    string pathPNG = basePath + ".png";
    
    cout << "Using RGB file: " << pathRGB << endl;
    cout << "Using PNG file: " << pathPNG << endl;

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
    
    unsigned char* buffer = ImageLoader::loadRawRGB(pathRGB, width, height);
    if (!buffer) return -1;

    // Load raw .rgb
    Mat rgbImage(height, width, CV_8UC3, buffer);

    Mat bgrImage;
    cvtColor(rgbImage, bgrImage, COLOR_RGB2BGR);

    // Debug: extract pieces and save to file
    string outputDir = "./extracted_pieces";
    vector<Mat> pieces = PieceExtractor::extractPieces(bgrImage, false, outputDir);
    
    // Also extract pieces with position info for animation
    vector<PieceInfo> pieceInfos = PieceExtractor::extractPiecesWithInfo(bgrImage, false, outputDir);

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

    cv::Mat contourImage = PieceExtractor::drawContoursOnImage(bgrImage);
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

    // Deprecated: use improved raster scan
    // vector<Pair> allMatches = PieceMatcher::createFilteredMatches(features, 0.9);
    // PuzzleLayout layout = PieceMatcher::buildLayout(allMatches, features, canvasW, canvasH);
    
    PuzzleLayout layout = PieceMatcher::buildLayoutRasterScan(features, canvasW, canvasH);

    // Debug: check if layout is successful
    if (layout.positions.empty()) {
        cerr << "ERROR: Layout building failed! No pieces placed." << endl;
        cerr << "Falling back to original method..." << endl;
        vector<Pair> allMatches = PieceMatcher::createFilteredMatches(features, 0.9);
        layout = PieceMatcher::buildLayout(allMatches, features, canvasW, canvasH);
    }
    
    if (layout.positions.empty()) {
        cerr << "ERROR: Still no pieces in layout! Cannot display results." << endl;
        return -1;
    }

    Mat finalAssembly(canvasH, canvasW, CV_8UC3, Scalar(0,0,0));

    // Calculate bounds properly
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    bool first = true;
    for (const auto& entry : layout.positions) {
        const auto& pos = entry.second;
        if (first) {
            minX = pos.position.x;
            minY = pos.position.y;
            maxX = pos.position.x + pos.size.width;
            maxY = pos.position.y + pos.size.height;
            first = false;
        } else {
            minX = min(minX, pos.position.x);
            minY = min(minY, pos.position.y);
            maxX = max(maxX, pos.position.x + pos.size.width);
            maxY = max(maxY, pos.position.y + pos.size.height);
        }
    }
    
    cout << "Layout bounds: [" << minX << ", " << minY << "] to [" << maxX << ", " << maxY << "]" << endl;

    // Calculate center offset
    float layoutWidth = maxX - minX;
    float layoutHeight = maxY - minY;
    float centerX = (canvasW - layoutWidth) / 2.0f;
    float centerY = (canvasH - layoutHeight) / 2.0f;

    int piecesDrawn = 0;
    for (auto& entry : layout.positions) {
        int pieceId = entry.first;
        const PiecePosition& pos = entry.second;
        
        if (pieceId < 0 || pieceId >= features.size()) {
            cerr << "WARNING: Invalid piece ID " << pieceId << endl;
            continue;
        }
        
        const Mat& piece = features[pieceId].img;
        Mat pieceToDraw = PieceMatcher::rotatePiece(piece, pos.rotation);

        int drawWidth = pos.size.width;
        int drawHeight = pos.size.height;
        
        if (drawWidth <= 0 || drawHeight <= 0) {
            cerr << "WARNING: Invalid size for piece " << pieceId << ": " << drawWidth << "x" << drawHeight << endl;
            continue;
        }

        // Center the entire layout
        int screenX = static_cast<int>(pos.position.x - minX + centerX);
        int screenY = static_cast<int>(pos.position.y - minY + centerY);

        // ensure within canvas
        if (screenX >= 0 && screenY >= 0 &&
            screenX + drawWidth <= finalAssembly.cols &&
            screenY + drawHeight <= finalAssembly.rows) {
            if (pieceToDraw.cols != drawWidth || pieceToDraw.rows != drawHeight) {
                resize(pieceToDraw, pieceToDraw, Size(drawWidth, drawHeight));
            }
            pieceToDraw.copyTo(finalAssembly(Rect(screenX, screenY, drawWidth, drawHeight)));
            piecesDrawn++;
        } else {
            int clipX = max(0, screenX);
            int clipY = max(0, screenY);
            int clipW = min(drawWidth, finalAssembly.cols - clipX);
            int clipH = min(drawHeight, finalAssembly.rows - clipY);
            if (clipW > 0 && clipH > 0) {
                Mat clippedPiece = pieceToDraw(Rect(0, 0, clipW, clipH));
                clippedPiece.copyTo(finalAssembly(Rect(clipX, clipY, clipW, clipH)));
                piecesDrawn++;
            } else {
                cerr << "WARNING: Piece " << pieceId << " completely outside canvas" << endl;
            }
        }

        putText(finalAssembly, to_string(pieceId), Point(screenX + 5, screenY + 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
        rectangle(finalAssembly, Point(screenX, screenY),
                Point(screenX + drawWidth, screenY + drawHeight), Scalar(0,255,0), 1);
    }
    
    cout << "Pieces drawn: " << piecesDrawn << "/" << layout.positions.size() << endl;

    // Prepare initial positions and rotations from the scrambled image
    vector<Point2f> initialPositions;
    vector<float> initialRotations;
    
    for (const auto& info : pieceInfos) {
        initialPositions.push_back(info.center);
        initialRotations.push_back(info.originalRotation);
    }

    // Configure and run animation
    PuzzleAnimator::AnimationConfig animConfig;
    animConfig.totalFrames = 180;     // 6 seconds at 30 FPS
    animConfig.fps = 30;
    animConfig.showWindow = true;
    animConfig.saveFrames = false;    // Set to true to save frames to disk

    cout << "\nStarting puzzle assembly animation from initial positions..." << endl;
    cout << "Initial positions: " << initialPositions.size() << " pieces" << endl;
    
    // Use the 5-parameter version that accepts initial positions
    PuzzleAnimator::animatePuzzleAssembly(features, layout, canvasW, canvasH, 
                                          initialPositions, initialRotations, animConfig);

    // Display final assembled puzzle and wait for keypress
    Mat finalPuzzle(canvasH, canvasW, CV_8UC3, Scalar(0, 0, 0));

    for (auto& entry : layout.positions) {
        int pieceId = entry.first;
        const PiecePosition& pos = entry.second;
        const Mat& piece = features[pieceId].img;
        Mat pieceToDraw = PieceMatcher::rotatePiece(piece, pos.rotation);

        int screenX = static_cast<int>(pos.position.x - minX + 50);
        int screenY = static_cast<int>(pos.position.y - minY + 50);

        if (screenX >= 0 && screenY >= 0 &&
            screenX + pieceToDraw.cols <= canvasW &&
            screenY + pieceToDraw.rows <= canvasH) {
            pieceToDraw.copyTo(finalPuzzle(Rect(screenX, screenY, pieceToDraw.cols, pieceToDraw.rows)));
        }
    }

    putText(finalPuzzle, "Animation Complete!", 
            Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
    putText(finalPuzzle, "Press any key to exit...", 
            Point(20, canvasH - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 2);

    imshow("Final Puzzle", finalPuzzle);
    waitKey(0);
    destroyWindow("Final Puzzle");

    cout << "Program completed successfully!" << endl;
    return 0;
}