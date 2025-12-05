#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace FeatureExtractor {

// use histogram method: extract RGB histogram feature of edge strip
// parameters: stripWidth - strip width (pixels), numBins - number of bins per channel (8 bins per channel)
static EdgeFeature extractEdgeHistogram(const Mat& img, int edge, int stripWidth = 4, int numBins = 8) {
    EdgeFeature ef;
    int h = img.rows;
    int w = img.cols;
    
    ef.histogram.resize(numBins * 3, 0.0);
    
    // collect all pixels of edge strip
    vector<Vec3b> stripPixels;
    
    int edgeLength = (edge == 0 || edge == 2) ? w : h;  // top/bottom use width, left/right use height
    
    for (int pos = 0; pos < edgeLength; pos++) {
        float t = float(pos) / max(1, edgeLength - 1);
        
        // collect strip pixels of stripWidth width according to edge direction
        for (int offset = 0; offset < stripWidth; offset++) {
            int x, y;
            switch (edge) {
                case 0: 
                    x = t * (w - 1);
                    y = max(0, min(h-1, offset));
                    break;
                case 1:
                    x = max(0, min(w-1, w-1 - offset));
                    y = t * (h - 1);
                    break;
                case 2:
                    x = t * (w - 1);
                    y = max(0, min(h-1, h-1 - offset));
                    break;
                default:
                    x = max(0, min(w-1, offset));
                    y = t * (h - 1);
                    break;
            }
            
            if (x >= 0 && x < w && y >= 0 && y < h) {
                Vec3b pixel = img.at<Vec3b>(y, x);
                stripPixels.push_back(pixel);
            }
        }
    }
    
    if (stripPixels.empty()) {
        return ef;  // if no pixels, return empty histogram
    }
    
    // calculate RGB three channels histogram
    // each bin range: 0-31, 32-63, 64-95, 96-127, 128-159, 160-191, 192-223, 224-255
    int binSize = 256 / numBins;
    
    for (const auto& pixel : stripPixels) {
        // B channel (pixel[0])
        int bBin = min(numBins - 1, pixel[0] / binSize);
        ef.histogram[bBin] += 1.0;
        
        // G channel (pixel[1])
        int gBin = min(numBins - 1, pixel[1] / binSize);
        ef.histogram[numBins + gBin] += 1.0;
        
        // R channel (pixel[2])
        int rBin = min(numBins - 1, pixel[2] / binSize);
        ef.histogram[2 * numBins + rBin] += 1.0;
    }
    
    // normalize histogram: each channel is normalized (the sum of each channel's bin is 1)
    double pixelCount = static_cast<double>(stripPixels.size());
    if (pixelCount > 0) {
        // B channel normalized [0, numBins-1]
        double bSum = 0.0;
        for (int i = 0; i < numBins; i++) {
            bSum += ef.histogram[i];
        }
        if (bSum > 0) {
            for (int i = 0; i < numBins; i++) {
                ef.histogram[i] /= bSum;
            }
        }
        
        // G channel normalized [numBins, 2*numBins-1]
        double gSum = 0.0;
        for (int i = numBins; i < 2 * numBins; i++) {
            gSum += ef.histogram[i];
        }
        if (gSum > 0) {
            for (int i = numBins; i < 2 * numBins; i++) {
                ef.histogram[i] /= gSum;
            }
        }
        
        // R channel normalized [2*numBins, 3*numBins-1]
        double rSum = 0.0;
        for (int i = 2 * numBins; i < 3 * numBins; i++) {
            rSum += ef.histogram[i];
        }
        if (rSum > 0) {
            for (int i = 2 * numBins; i < 3 * numBins; i++) {
                ef.histogram[i] /= rSum;
            }
        }
    }
    
    return ef;
}

static EdgeFeature extractEdgeRGBProfile(const Mat& img, int edge, int stripWidth = 5, 
                                         int numSamples = 60, int numBins = 10) {
    EdgeFeature ef;
    int h = img.rows;
    int w = img.cols;
    
    int edgeLength = (edge == 0 || edge == 2) ? w : h;
    
    // adaptive sampling points number (adjust according to edge length)
    // ensure small pieces have enough sampling points, large pieces will not be oversampled
    int adaptiveSamples = max(40, min(80, static_cast<int>(edgeLength * 0.15)));
    numSamples = adaptiveSamples;
    
    // 1. extract RGB sampling points (preserve spatial order, for shape matching)
    ef.rgbVals.resize(numSamples);
    ef.vals.resize(numSamples);  // save grayscale value for backward compatibility
    
    // 2. collect strip pixels for histogram
    vector<Vec3b> stripPixels;
    stripPixels.reserve(edgeLength * stripWidth);
    
    // initialize histogram
    ef.histogram.resize(numBins * 3, 0.0);
    
    for (int i = 0; i < numSamples; i++) {
        float t = float(i) / (numSamples - 1);
        double sumR = 0.0, sumG = 0.0, sumB = 0.0, sumGray = 0.0;
        int count = 0;
        
        // sample in stripWidth width
        for (int offset = 0; offset < stripWidth; offset++) {
            int x, y;
            switch (edge) {
                case 0:  // top
                    x = t * (w - 1);
                    y = max(0, min(h-1, offset));
                    break;
                case 1:  // right
                    x = max(0, min(w-1, w-1 - offset));
                    y = t * (h - 1);
                    break;
                case 2:  // bottom
                    x = t * (w - 1);
                    y = max(0, min(h-1, h-1 - offset));
                    break;
                default: // left
                    x = max(0, min(w-1, offset));
                    y = t * (h - 1);
                    break;
            }
            
            if (x >= 0 && x < w && y >= 0 && y < h) {
                Vec3b pixel = img.at<Vec3b>(y, x);
                sumB += pixel[0];
                sumG += pixel[1];
                sumR += pixel[2];
                sumGray += 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
                count++;
                
                stripPixels.push_back(pixel);
            }
        }
        
        if (count > 0) {
            double avgB = sumB / count;
            double avgG = sumG / count;
            double avgR = sumR / count;
            
            if (i > 0 && ef.rgbVals[i-1][0] >= 0) {
                Vec3d prev = ef.rgbVals[i-1];
                double diffB = abs(avgB - prev[0]);
                double diffG = abs(avgG - prev[1]);
                double diffR = abs(avgR - prev[2]);
                
                if (diffB > 50 || diffG > 50 || diffR > 50) {
                    avgB = 0.7 * avgB + 0.3 * prev[0];
                    avgG = 0.7 * avgG + 0.3 * prev[1];
                    avgR = 0.7 * avgR + 0.3 * prev[2];
                }
            }
            
            ef.rgbVals[i] = Vec3d(avgB, avgG, avgR);
            ef.vals[i] = sumGray / count;
        }
    }
    
    // 3. calculate RGB histogram (statistical feature, robust to noise)
    if (!stripPixels.empty()) {
        int binSize = 256 / numBins;
        double totalWeight = 0.0;
        
        for (size_t idx = 0; idx < stripPixels.size(); idx++) {
            const auto& pixel = stripPixels[idx];
            
            int posInStrip = idx % stripWidth;
            double weight = exp(-0.5 * pow((posInStrip - stripWidth/2.0) / (stripWidth/3.0), 2));
            totalWeight += weight;
            
            int bBin = min(numBins - 1, pixel[0] / binSize);
            int gBin = min(numBins - 1, pixel[1] / binSize);
            int rBin = min(numBins - 1, pixel[2] / binSize);
            
            ef.histogram[bBin] += weight;
            ef.histogram[numBins + gBin] += weight;
            ef.histogram[2 * numBins + rBin] += weight;
        }
        
        double bSum = 0, gSum = 0, rSum = 0;
        for (int i = 0; i < numBins; i++) {
            bSum += ef.histogram[i];
            gSum += ef.histogram[numBins + i];
            rSum += ef.histogram[2 * numBins + i];
        }
        if (bSum > 1e-6) for (int i = 0; i < numBins; i++) ef.histogram[i] /= bSum;
        if (gSum > 1e-6) for (int i = numBins; i < 2*numBins; i++) ef.histogram[i] /= gSum;
        if (rSum > 1e-6) for (int i = 2*numBins; i < 3*numBins; i++) ef.histogram[i] /= rSum;
    }
    
    return ef;
}

static Mat preprocessPiece(const Mat& piece) {
    Mat processed = piece.clone();
    
    // 1. slight Gaussian blur to remove noise, preserve edge information
    GaussianBlur(processed, processed, Size(3, 3), 0.5);
    
    // 2. contrast enhancement
    vector<Mat> channels;
    split(processed, channels);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(channels[0], channels[0]);
    clahe->apply(channels[1], channels[1]);
    clahe->apply(channels[2], channels[2]);
    merge(channels, processed);
    
    return processed;
}

PieceFeature extract(const Mat& piece) {
    PieceFeature pf;
    pf.img = piece.clone();

    Mat processedPiece = preprocessPiece(piece);

    int stripWidth = 5;
    int numSamples = 60;
    int numBins = 10;

    pf.top    = extractEdgeRGBProfile(processedPiece, 0, stripWidth, numSamples, numBins);
    pf.right  = extractEdgeRGBProfile(processedPiece, 1, stripWidth, numSamples, numBins);
    pf.bottom = extractEdgeRGBProfile(processedPiece, 2, stripWidth, numSamples, numBins);
    pf.left   = extractEdgeRGBProfile(processedPiece, 3, stripWidth, numSamples, numBins);

    return pf;
}



}
