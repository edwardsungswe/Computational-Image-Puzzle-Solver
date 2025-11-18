#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

#include "FeatureExtractor.h"
#include "Matcher.h"
#include "Assembler.h"

using namespace std;
using namespace cv;

unsigned char* readImageData(const string& imagePath, int width, int height) {
    ifstream inputFile(imagePath, ios::binary);
    if (!inputFile.is_open()) {
        cerr << "Error Opening File for Reading: " << imagePath << endl;
        exit(1);
    }

    vector<char> Rbuf(width * height);
    vector<char> Gbuf(width * height);
    vector<char> Bbuf(width * height);

    inputFile.read(Rbuf.data(), width * height);
    inputFile.read(Gbuf.data(), width * height);
    inputFile.read(Bbuf.data(), width * height);
    inputFile.close();

    unsigned char* inData = (unsigned char*)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        inData[3 * i]     = Rbuf[i];
        inData[3 * i + 1] = Gbuf[i];
        inData[3 * i + 2] = Bbuf[i];
    }

#include "ImageIO/ImageLoader.h"
#include "Pieces/PieceExtractor.h"

int main() {
    string path     = "../data_sample/starry_night_rotate.rgb";
    string pathPNG  = "../data_sample/starry_night_rotate.png";

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
    unsigned char* buffer = readImageData(path, width, height);
    Mat rgbImage(height, width, CV_8UC3, buffer);

    // Convert & mask
    Mat gray;
    cvtColor(rgbImage, gray, COLOR_RGB2GRAY);

    Mat mask;
    threshold(gray, mask, 10, 255, THRESH_BINARY);

    // Contours
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = rgbImage.clone();
    vector<Mat> pieces;

    for (const auto& c : contours) {
        RotatedRect box = minAreaRect(c);

        // Draw box
        Point2f ptsf[4];
        box.points(ptsf);
        vector<Point> poly;
        for (int i = 0; i < 4; i++) {
            poly.emplace_back(Point(cvRound(ptsf[i].x), cvRound(ptsf[i].y)));
        }
        polylines(contourImage, poly, true, Scalar(0,0,255), 2);

        float angle = box.angle;
        Size2f boxSize = box.size;
        if (box.angle < -45.0f) {
            angle += 90.0f;
            swap(boxSize.width, boxSize.height);
        }

        // Rotate
        Mat M = getRotationMatrix2D(box.center, angle, 1.0);
        Mat rotated;
        warpAffine(rgbImage, rotated, M, rgbImage.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0,0,0));

        // ROI extraction
        Rect roi;
        roi.width  = (int)boxSize.width;
        roi.height = (int)boxSize.height;
        roi.x = (int)(box.center.x - roi.width  / 2);
        roi.y = (int)(box.center.y - roi.height / 2);

        roi.x = max(0, min(roi.x, rotated.cols - 1));
        roi.y = max(0, min(roi.y, rotated.rows - 1));
        if (roi.x + roi.width > rotated.cols)  roi.width  = rotated.cols - roi.x;
        if (roi.y + roi.height > rotated.rows) roi.height = rotated.rows - roi.y;

        if (roi.width > 0 && roi.height > 0) {
            pieces.push_back(rotated(roi).clone());
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

    showStage(rgbImage, "Stage 1: Original Image");

    showStage(contourImage, "Stage 2: Detected Contours");

    std::cout << "Press any key..." << std::endl;
    cv::waitKey(0);

    Mat preview(canvasH, canvasW, CV_8UC3, Scalar(0,0,0));

    int x = 10, y = 10, rowH = 0;

    for (auto& piece : pieces) {
        if (x + piece.cols > canvasW - 10) {
            x = 10;
            y += rowH + 10;
            rowH = 0;
        }
        if (y + piece.rows > canvasH - 10) break;

        piece.copyTo(preview(Rect(x, y, piece.cols, piece.rows)));
        x += piece.cols + 10;
        rowH = max(rowH, piece.rows);
    }

    showStage(preview, "Stage 3: Corrected Pieces");

    vector<PieceFeature> features;
    for (auto& p : pieces) features.push_back(FeatureExtractor::extract(p));

    vector<pair<int,double>> RL = Matcher::matchAll(features);
    PuzzleLayout layout = Matcher::buildLayout(features.size());
    vector<Point> targetPos = Assembler::computePiecePositions(layout, features, canvasW, canvasH);

    vector<Point> startPos(features.size());
    for (int i = 0; i < features.size(); i++) {
        startPos[i] = Point(rand() % canvasW, rand() % canvasH);
    }

    canvas = Mat(canvasH, canvasW, CV_8UC3, Scalar(0,0,0));
    putText(canvas, "Stage 4: Animation (auto play)",
            Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);
    putText(canvas, "Press any key to start animation...",
            Point(20, canvasH - 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200,200,200), 2);
    imshow("Puzzle Demo", canvas);
    waitKey(0);

    for (int frame = 0; frame < 60; frame++) {
        float t = frame / 59.0f;
        canvas = Mat(canvasH, canvasW, CV_8UC3, Scalar(0,0,0));

        for (int i = 0; i < features.size(); i++) {
            int x = startPos[i].x * (1 - t) + targetPos[i].x * t;
            int y = startPos[i].y * (1 - t) + targetPos[i].y * t;

            const Mat& piece = features[i].img;
            if (x >= 0 && y >= 0 &&
                x + piece.cols <= canvasW &&
                y + piece.rows <= canvasH)
            {
                piece.copyTo(canvas(Rect(x, y, piece.cols, piece.rows)));
            }
        }

        putText(canvas, "Stage 4: Animation",
                Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);

        imshow("Puzzle Demo", canvas);
        waitKey(30);
    }

    Mat finalImage = Assembler::assembleImage(layout, features, canvasW, canvasH);

    showStage(finalImage, "Final Result (Press any key to exit)");
    return 0;
}
