#include "FeatureExtractor.h"
#include <iostream>

using namespace cv;
using namespace std;

namespace FeatureExtractor {

EdgeFeature extractEdge(const Mat& img, int edge) {
    EdgeFeature ef;
    
    if (img.empty()) {
        ef.length = 0;
        return ef;
    }

    int h = img.rows;
    int w = img.cols;

    // Compute gradients for texture matching
    Mat gray, gradX, gradY;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Sobel(gray, gradX, CV_64F, 1, 0, 3);
    Sobel(gray, gradY, CV_64F, 0, 1, 3);

    // Extract boundary pixels and gradients based on edge
    switch (edge) {
        case 0:  // Top edge - first row, left to right
            ef.length = w;
            ef.boundaryPixels.reserve(w);
            ef.gradients.reserve(w);
            for (int x = 0; x < w; x++) {
                ef.boundaryPixels.push_back(img.at<Vec3b>(0, x));
                // Use vertical gradient for horizontal edges
                ef.gradients.push_back(gradY.at<double>(0, x));
            }
            break;

        case 1:  // Right edge - last column, top to bottom
            ef.length = h;
            ef.boundaryPixels.reserve(h);
            ef.gradients.reserve(h);
            for (int y = 0; y < h; y++) {
                ef.boundaryPixels.push_back(img.at<Vec3b>(y, w - 1));
                // Use horizontal gradient for vertical edges
                ef.gradients.push_back(gradX.at<double>(y, w - 1));
            }
            break;

        case 2:  // Bottom edge - last row, left to right
            ef.length = w;
            ef.boundaryPixels.reserve(w);
            ef.gradients.reserve(w);
            for (int x = 0; x < w; x++) {
                ef.boundaryPixels.push_back(img.at<Vec3b>(h - 1, x));
                ef.gradients.push_back(gradY.at<double>(h - 1, x));
            }
            break;

        case 3:  // Left edge - first column, top to bottom
            ef.length = h;
            ef.boundaryPixels.reserve(h);
            ef.gradients.reserve(h);
            for (int y = 0; y < h; y++) {
                ef.boundaryPixels.push_back(img.at<Vec3b>(y, 0));
                ef.gradients.push_back(gradX.at<double>(y, 0));
            }
            break;
    }

    return ef;
}

PieceFeature extract(const Mat& piece) {
    PieceFeature pf;

    if (piece.empty() || piece.channels() != 3) {
        cerr << "Warning: Invalid piece image" << endl;
        return pf;
    }

    pf.img = piece.clone();
    pf.size = piece.size();

    // Extract all four edges
    pf.top = extractEdge(piece, 0);
    pf.right = extractEdge(piece, 1);
    pf.bottom = extractEdge(piece, 2);
    pf.left = extractEdge(piece, 3);

    return pf;
}

}