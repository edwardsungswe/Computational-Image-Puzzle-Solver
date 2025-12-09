#include "PieceExtractor.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

using namespace std;
using namespace cv;

namespace {

void normalizeRotatedRect(float& angle, Size2f& size) {
    while (angle < -45.0f) { angle += 90.0f; swap(size.width, size.height); }
    while (angle > 45.0f)  { angle -= 90.0f; swap(size.width, size.height); }
}

Rect computeSafeROI(Point2f center, Size2f size, Size imageSize) {
    int halfW = static_cast<int>(ceil(size.width / 2.0f));
    int halfH = static_cast<int>(ceil(size.height / 2.0f));

    int x = static_cast<int>(round(center.x)) - halfW;
    int y = static_cast<int>(round(center.y)) - halfH;
    int w = static_cast<int>(ceil(size.width));
    int h = static_cast<int>(ceil(size.height));

    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > imageSize.width)  w = imageSize.width - x;
    if (y + h > imageSize.height) h = imageSize.height - y;

    return Rect(x, y, max(1, w), max(1, h));
}

void sortContoursByArea(vector<vector<Point>>& contours) {
    sort(contours.begin(), contours.end(),
         [](const vector<Point>& a, const vector<Point>& b) {
             return contourArea(a) > contourArea(b);
         });
}

}

vector<Mat> PieceExtractor::extractPieces(const Mat& bgrImage, const string& outputDir) {
    vector<Mat> pieces;

    if (bgrImage.empty()) {
        cerr << "Empty image passed to extractPieces" << endl;
        return pieces;
    }

    bool savePieces = !outputDir.empty();
    if (savePieces) {
        filesystem::create_directories(outputDir);
    }

    Mat gray, mask;
    cvtColor(bgrImage, gray, COLOR_BGR2GRAY);
    threshold(gray, mask, 15, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    sortContoursByArea(contours);

    int pieceIndex = 0;
    const double MIN_AREA = 100;

    for (const auto& contour : contours) {
        if (contourArea(contour) < MIN_AREA) continue;

        RotatedRect box = minAreaRect(contour);
        float angle = box.angle;
        Size2f boxSize = box.size;
        normalizeRotatedRect(angle, boxSize);

        if (boxSize.width < 10 || boxSize.height < 10) continue;

        Mat piece;

        if (abs(angle) > 1.0f) {
            Mat rotationMatrix = getRotationMatrix2D(box.center, angle, 1.0);
            Mat rotated;
            warpAffine(bgrImage, rotated, rotationMatrix, bgrImage.size(), INTER_CUBIC, BORDER_REPLICATE);

            Rect roi = computeSafeROI(box.center, boxSize, rotated.size());
            if (roi.width <= 0 || roi.height <= 0) continue;
            piece = rotated(roi).clone();
        } else {
            Rect bbox = box.boundingRect() & Rect(0, 0, bgrImage.cols, bgrImage.rows);
            if (bbox.width <= 0 || bbox.height <= 0) continue;
            piece = bgrImage(bbox).clone();
        }

        Scalar mean = cv::mean(piece);
        if (mean[0] + mean[1] + mean[2] < 30) continue;

        pieces.push_back(piece);

        if (savePieces) {
            imwrite(outputDir + "/piece_" + to_string(pieceIndex) + ".png", piece);
        }
        pieceIndex++;
    }

    return pieces;
}

vector<PieceInfo> PieceExtractor::extractPiecesWithInfo(const Mat& bgrImage, const string& outputDir) {
    vector<PieceInfo> pieces;

    if (bgrImage.empty()) {
        cerr << "Empty image passed to extractPiecesWithInfo" << endl;
        return pieces;
    }

    bool savePieces = !outputDir.empty();
    if (savePieces) {
        filesystem::create_directories(outputDir);
    }

    Mat gray, mask;
    cvtColor(bgrImage, gray, COLOR_BGR2GRAY);
    threshold(gray, mask, 15, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    sortContoursByArea(contours);

    int pieceIndex = 0;
    const double MIN_AREA = 100;

    for (const auto& contour : contours) {
        if (contourArea(contour) < MIN_AREA) continue;

        RotatedRect box = minAreaRect(contour);
        float angle = box.angle;
        Size2f boxSize = box.size;
        normalizeRotatedRect(angle, boxSize);

        if (boxSize.width < 10 || boxSize.height < 10) continue;

        Mat pieceImg;

        if (abs(angle) > 1.0f) {
            Mat rotationMatrix = getRotationMatrix2D(box.center, angle, 1.0);
            Mat rotated;
            warpAffine(bgrImage, rotated, rotationMatrix, bgrImage.size(), INTER_CUBIC, BORDER_REPLICATE);

            Rect roi = computeSafeROI(box.center, boxSize, rotated.size());
            if (roi.width <= 0 || roi.height <= 0) continue;
            pieceImg = rotated(roi).clone();
        } else {
            Rect bbox = box.boundingRect() & Rect(0, 0, bgrImage.cols, bgrImage.rows);
            if (bbox.width <= 0 || bbox.height <= 0) continue;
            pieceImg = bgrImage(bbox).clone();
        }

        Scalar mean = cv::mean(pieceImg);
        if (mean[0] + mean[1] + mean[2] < 30) continue;

        PieceInfo info;
        info.img = pieceImg;
        info.center = box.center;
        info.originalRotation = angle;
        info.originalSize = Size(static_cast<int>(boxSize.width), static_cast<int>(boxSize.height));
        pieces.push_back(info);

        if (savePieces) {
            imwrite(outputDir + "/piece_" + to_string(pieceIndex) + ".png", pieceImg);
        }
        pieceIndex++;
    }

    return pieces;
}

Mat PieceExtractor::drawContoursOnImage(const Mat& bgrImage) {
    Mat gray, mask;
    cvtColor(bgrImage, gray, COLOR_BGR2GRAY);
    threshold(gray, mask, 15, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result = bgrImage.clone();
    int idx = 0;

    for (const auto& contour : contours) {
        if (contourArea(contour) < 100) continue;

        RotatedRect box = minAreaRect(contour);
        Point2f pts[4];
        box.points(pts);

        for (int i = 0; i < 4; i++) {
            line(result, pts[i], pts[(i + 1) % 4], Scalar(0, 0, 255), 2);
        }

        putText(result, to_string(idx), box.center, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        idx++;
    }

    return result;
}
