#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace FeatureExtractor {

static EdgeFeature sampleEdgeFromMat(const Mat& Y, int edge, int nSamples) {
    EdgeFeature ef;
    int h = Y.rows;
    int w = Y.cols;
    ef.vals.resize(nSamples);

    int sampleWidth = 15;
    for (int i = 0; i < nSamples; i++) {
        float t = float(i) / (nSamples - 1);
        double sum = 0.0;
        int count = 0;

        for (int offset = -sampleWidth/2; offset <= sampleWidth/2; offset++) {
            int x, y;
            switch (edge) {
                case 0: 
                    x = t * (w - 1); 
                    y = max(0, min(h-1, offset)); 
                    break;
                case 1: 
                    x = max(0, min(w-1, w-1 + offset)); 
                    y = t * (h - 1); 
                    break;
                case 2: 
                    x = t * (w - 1); 
                    y = max(0, min(h-1, h-1 + offset)); 
                    break;
                default: 
                    x = max(0, min(w-1, offset)); 
                    y = t * (h - 1); 
                    break;
            }
            sum += Y.at<uchar>(y, x);
            count++;
        }
        ef.vals[i] = sum / count;
    }
    return ef;
}

PieceFeature extract(const Mat& piece) {
    PieceFeature pf;
    pf.img = piece.clone();

    Mat yuv;
    cvtColor(piece, yuv, COLOR_BGR2YUV);
    vector<Mat> c;
    split(yuv, c);
    Mat Y = c[0];

    int N = 40;

    pf.top    = sampleEdgeFromMat(Y, 0, N);
    pf.right  = sampleEdgeFromMat(Y, 1, N);
    pf.bottom = sampleEdgeFromMat(Y, 2, N);
    pf.left   = sampleEdgeFromMat(Y, 3, N);

    return pf;
}

}
