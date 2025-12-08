#include "Assembler.h"
#include <iostream>

using namespace std;
using namespace cv;

namespace {

Mat rotateImage90(const Mat& img, int steps) {
    steps = ((steps % 4) + 4) % 4;
    if (steps == 0) return img.clone();

    Mat result;
    if (steps == 1) { transpose(img, result); flip(result, result, 0); }
    else if (steps == 2) { flip(img, result, -1); }
    else { transpose(img, result); flip(result, result, 1); }
    return result;
}

}

namespace Assembler {

Mat assemble(const vector<PieceFeature>& features, const PuzzleLayout& layout) {
    if (layout.positions.empty()) {
        cerr << "Empty layout" << endl;
        return Mat();
    }

    int outputW = static_cast<int>(ceil(layout.bounds.width));
    int outputH = static_cast<int>(ceil(layout.bounds.height));

    return assemble(features, layout, outputW, outputH);
}

Mat assemble(const vector<PieceFeature>& features, const PuzzleLayout& layout,
             int outputWidth, int outputHeight) {

    if (layout.positions.empty()) {
        cerr << "Empty layout" << endl;
        return Mat();
    }

    Mat canvas = Mat::zeros(outputHeight, outputWidth, CV_8UC3);

    for (const auto& entry : layout.positions) {
        int pieceId = entry.first;
        const PiecePosition& pos = entry.second;

        if (pieceId < 0 || pieceId >= static_cast<int>(features.size())) continue;

        Mat piece = features[pieceId].img;
        if (piece.empty()) continue;

        int rotSteps = static_cast<int>(round(pos.rotation / 90.0f)) % 4;
        if (rotSteps < 0) rotSteps += 4;
        Mat rotatedPiece = rotateImage90(piece, rotSteps);

        int x = static_cast<int>(round(pos.position.x));
        int y = static_cast<int>(round(pos.position.y));
        int w = rotatedPiece.cols;
        int h = rotatedPiece.rows;

        int srcX = 0, srcY = 0;
        int dstX = x, dstY = y;
        int copyW = w, copyH = h;

        if (dstX < 0) { srcX = -dstX; copyW += dstX; dstX = 0; }
        if (dstY < 0) { srcY = -dstY; copyH += dstY; dstY = 0; }
        if (dstX + copyW > outputWidth)  copyW = outputWidth - dstX;
        if (dstY + copyH > outputHeight) copyH = outputHeight - dstY;

        if (copyW <= 0 || copyH <= 0) continue;

        Rect srcROI(srcX, srcY, copyW, copyH);
        Rect dstROI(dstX, dstY, copyW, copyH);
        rotatedPiece(srcROI).copyTo(canvas(dstROI));
    }

    return canvas;
}

Mat assembleWithDebug(const vector<PieceFeature>& features, const PuzzleLayout& layout) {
    Mat result = assemble(features, layout);
    if (result.empty()) return result;

    for (const auto& entry : layout.positions) {
        int pieceId = entry.first;
        const PiecePosition& pos = entry.second;

        int x = static_cast<int>(pos.position.x);
        int y = static_cast<int>(pos.position.y);
        int w = pos.size.width;
        int h = pos.size.height;

        rectangle(result, Rect(x, y, w, h), Scalar(0, 255, 0), 1);

        string label = to_string(pieceId) + "(" + to_string(static_cast<int>(pos.rotation)) + ")";
        int baseline;
        Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
        Point textPos(x + w / 2 - textSize.width / 2, y + h / 2 + textSize.height / 2);

        rectangle(result,
                  Point(textPos.x - 2, textPos.y - textSize.height - 2),
                  Point(textPos.x + textSize.width + 2, textPos.y + 2),
                  Scalar(0, 0, 0), -1);
        putText(result, label, textPos, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);

        Point center(x + w / 2, y + h / 2);
        int arrowLen = min(w, h) / 4;
        double angleRad = pos.rotation * CV_PI / 180.0;
        Point arrowEnd(
            center.x - static_cast<int>(arrowLen * sin(angleRad)),
            center.y - static_cast<int>(arrowLen * cos(angleRad))
        );
        arrowedLine(result, center, arrowEnd, Scalar(255, 0, 0), 2, LINE_AA, 0, 0.3);
    }

    string gridInfo = "Grid: " + to_string(layout.rows) + "x" + to_string(layout.cols);
    putText(result, gridInfo, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

    return result;
}

}