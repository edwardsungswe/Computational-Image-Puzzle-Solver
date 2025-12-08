#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Features/FeatureExtractor.h"
#include "Matching/PieceMatcher.h"
#include "ImageIO/ImageLoader.h"
#include "Pieces/PieceExtractor.h"

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
    vector<Mat> pieces = PieceExtractor::extractPieces(bgrImage, outputDir);

    vector<PieceFeature> features;
    for (const auto& p : pieces) {
        features.push_back(FeatureExtractor::extract(p));
    }

    PuzzleLayout layout = PieceMatcher::solve(features, width, height);

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
            pieceToDraw(srcROI).copyTo(finalPuzzle(dstROI));
        }
    }

    imwrite("output.png", finalPuzzle);
    imshow("Solved Puzzle", finalPuzzle);
    waitKey(0);
    destroyAllWindows();

    free(buffer);
    return 0;
}