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

// vector<Mat> PieceExtractor::extractPieces(const Mat& bgrImage, const string& outputDir) {
//     vector<Mat> pieces;    

//     if (bgrImage.empty()) {
//         cerr << "Empty image passed to extractPieces" << endl;
//         return pieces;
//     }

//     bool savePieces = !outputDir.empty();
//     if (savePieces) {
//         filesystem::create_directories(outputDir);
//     }    
    
//     Mat gray;
//     cvtColor(bgrImage, gray, COLOR_BGR2GRAY);

//     Mat mask;
//     threshold(gray, mask, 15, 255, THRESH_BINARY);

//     Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//     morphologyEx(mask, mask, MORPH_CLOSE, kernel);
//     morphologyEx(mask, mask, MORPH_OPEN, kernel);

//     // Find contours
//     vector<vector<Point>> contours;
//     findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//     sortContoursByArea(contours);

//     int pieceIndex = 0;
//     const double MIN_AREA = 100;
    
//     // For each contour, extract rotated piece
//     for (const auto& c : contours) {
//         if (contourArea(c) < MIN_AREA) continue;

//         RotatedRect rotatedBox = minAreaRect(c);
//         float angle = rotatedBox.angle;
//         Size2f boxSize = rotatedBox.size;

//         if (boxSize.width < 10 || boxSize.height < 10) continue;

//         normalizeRotatedRect(angle, boxSize);
//         Mat piece;
        
//         if (abs(angle) > 0.5f) {
//             Mat M = getRotationMatrix2D(rotatedBox.center, angle, 1.0);
//             Mat rotated;
//             warpAffine(bgrImage, rotated, M, bgrImage.size(), INTER_LINEAR, BORDER_CONSTANT);

//             Rect roi = computeSafeROI(rotatedBox.center, boxSize, rotated.size());
//             // cv::Rect roi;
//             // roi.width  = (int)boxSize.width;
//             // roi.height = (int)boxSize.height;
//             // roi.x = (int)(rotatedBox.center.x - roi.width  / 2);
//             // roi.y = (int)(rotatedBox.center.y - roi.height / 2);
//             roi.x += 4; roi.y += 4;  // Inset by a few pixels
//             roi.width -= 4; roi.height -= 4;
//             if (roi.width <= 0 || roi.height <= 0) continue;
            
//             piece = rotated(roi).clone();
//         }
//         else {
//             Rect bbox = boundingRect(c);
//             bbox.x = max(0, bbox.x);
//             bbox.y = max(0, bbox.y);
//             bbox.width = min(bgrImage.cols - bbox.x, bbox.width);
//             bbox.height = min(bgrImage.rows - bbox.y, bbox.height);
//             if (bbox.width <= 0 || bbox.height <= 0) continue;
//             piece = bgrImage(bbox).clone();
//         }

//         Scalar mean = cv::mean(piece);
//         if (mean[0] + mean[1] + mean[2] < 30) continue;

//         pieces.push_back(piece);

//         if (savePieces) {
//             imwrite(outputDir + "/piece_" + to_string(pieceIndex) + ".png", piece);
//         }        
//         pieceIndex++;
//     }
//     return pieces;
// }
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
    
    Mat gray;
    cvtColor(bgrImage, gray, COLOR_BGR2GRAY);
    Mat mask;
    threshold(gray, mask, 15, 255, THRESH_BINARY);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    sortContoursByArea(contours);
    
    int pieceIndex = 0;
    const double MIN_AREA = 100;
    
    for (const auto& c : contours) {
        if (contourArea(c) < MIN_AREA) continue;
        
        RotatedRect rotatedBox = minAreaRect(c);
        float angle = rotatedBox.angle;
        Size2f boxSize = rotatedBox.size;
        
        if (boxSize.width < 10 || boxSize.height < 10) continue;
        normalizeRotatedRect(angle, boxSize);
        
        Mat piece;
        
        if (abs(angle) > 0.5f) {
            Mat M = getRotationMatrix2D(rotatedBox.center, angle, 1.0);
            Mat rotated;
            warpAffine(bgrImage, rotated, M, bgrImage.size(), INTER_LINEAR, BORDER_REPLICATE);

            Rect roi = computeSafeROI(rotatedBox.center, boxSize, rotated.size());
            // cv::Rect roi;
            // roi.width  = (int)boxSize.width;
            // roi.height = (int)boxSize.height;
            // roi.x = (int)(rotatedBox.center.x - roi.width  / 2);
            // roi.y = (int)(rotatedBox.center.y - roi.height / 2);
            if (roi.width <= 0 || roi.height <= 0) continue;
            
            piece = rotated(roi).clone();
        }
        else {
            Rect bbox = boundingRect(c);
            bbox.x = max(0, bbox.x);
            bbox.y = max(0, bbox.y);
            bbox.width = min(bgrImage.cols - bbox.x, bbox.width);
            bbox.height = min(bgrImage.rows - bbox.y, bbox.height);
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

Mat removeRemainingEdgeArtifactsLight(const Mat& piece, int threshold) {
    if (piece.empty()) return piece.clone();
    
    Mat result = piece.clone();
    
    const int EDGE = 2;
    
    Mat gray;
    if (piece.channels() == 3) {
        cvtColor(piece, gray, COLOR_BGR2GRAY);
    } else {
        gray = piece.clone();
    }
    
    if (piece.rows < 10 || piece.cols < 10) return piece.clone();
    
    vector<Rect> edgeRegions = {
        Rect(0, 0, piece.cols, EDGE),
        Rect(0, piece.rows - EDGE, piece.cols, EDGE),
        Rect(0, 0, EDGE, piece.rows),
        Rect(piece.cols - EDGE, 0, EDGE, piece.rows)
    };
    
    for (const auto& edgeRect : edgeRegions) {
        Mat edgeRegion = gray(edgeRect);
        Mat darkPixels = edgeRegion < threshold;
        
        if (countNonZero(darkPixels) > edgeRect.area() * 0.8) {
            Mat edgeMask = Mat::zeros(piece.size(), CV_8UC1);
            edgeMask(edgeRect).setTo(255, darkPixels);
            
            if (piece.channels() == 3) {
                inpaint(result, edgeMask, result, 2, INPAINT_TELEA);
            }
        }
    }
    
    return result;
}

Mat cleanExtractionArtifacts(const Mat& piece, int sensitivity) {
    if (piece.empty()) return piece.clone();
    
    Mat result = piece.clone();
    
    Mat gray;
    if (piece.channels() == 3) {
        cvtColor(piece, gray, COLOR_BGR2GRAY);
    } else {
        gray = piece.clone();
    }
    
    Mat darkMask;
    threshold(gray, darkMask, sensitivity, 255, THRESH_BINARY_INV);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(darkMask, darkMask, MORPH_CLOSE, kernel);
    morphologyEx(darkMask, darkMask, MORPH_OPEN, kernel);
    
    vector<vector<Point>> darkContours;
    findContours(darkMask, darkContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : darkContours) {
        double area = contourArea(contour);
        Rect bbox = boundingRect(contour);
        
        bool isSmall = area < 300;
        bool isNearBorder = (bbox.x < 5 || bbox.y < 5 ||
                           bbox.x + bbox.width > piece.cols - 5 || 
                           bbox.y + bbox.height > piece.rows - 5);
        
        Mat region = gray(bbox);
        Scalar regionMean = mean(region);
        bool isVeryDark = regionMean[0] < 15;
        
        if (isSmall && isNearBorder && isVeryDark) {
            Mat artifactMask = Mat::zeros(piece.size(), CV_8UC1);
            drawContours(artifactMask, vector<vector<Point>>{contour}, 0, Scalar(255), -1);
            
            if (piece.channels() == 3) {
                inpaint(result, artifactMask, result, 3, INPAINT_TELEA);
            } else {
                Mat maskInverted = 255 - artifactMask;
                Scalar meanColor = mean(result, maskInverted);
                result.setTo(meanColor, artifactMask);
            }
        }
    }
    
    const int EDGE_CHECK_WIDTH = 7;
    
    int topBlackRows = 0;
    for (int y = 0; y < min(EDGE_CHECK_WIDTH, result.rows); y++) {
        Mat row = result.row(y);
        Scalar rowMean = mean(row);
        if (rowMean[0] < 10 && rowMean[1] < 10 && rowMean[2] < 10) {
            topBlackRows++;
        } else {
            break;
        }
    }
    
    int bottomBlackRows = 0;
    for (int y = result.rows - 1; y >= max(result.rows - EDGE_CHECK_WIDTH, 0); y--) {
        Mat row = result.row(y);
        Scalar rowMean = mean(row);
        if (rowMean[0] < 10 && rowMean[1] < 10 && rowMean[2] < 10) {
            bottomBlackRows++;
        } else {
            break;
        }
    }
    
    int leftBlackCols = 0;
    for (int x = 0; x < min(EDGE_CHECK_WIDTH, result.cols); x++) {
        Mat col = result.col(x);
        Scalar colMean = mean(col);
        if (colMean[0] < 10 && colMean[1] < 10 && colMean[2] < 10) {
            leftBlackCols++;
        } else {
            break;
        }
    }
    
    int rightBlackCols = 0;
    for (int x = result.cols - 1; x >= max(result.cols - EDGE_CHECK_WIDTH, 0); x--) {
        Mat col = result.col(x);
        Scalar colMean = mean(col);
        if (colMean[0] < 10 && colMean[1] < 10 && colMean[2] < 10) {
            rightBlackCols++;
        } else {
            break;
        }
    }
    
    if (topBlackRows > 2 || bottomBlackRows > 2 || leftBlackCols > 2 || rightBlackCols > 2) {
        int newX = leftBlackCols;
        int newY = topBlackRows;
        int newWidth = result.cols - leftBlackCols - rightBlackCols;
        int newHeight = result.rows - topBlackRows - bottomBlackRows;
        
        if (newWidth > piece.cols * 0.7 && newHeight > piece.rows * 0.7) {
            Rect cropRect(newX, newY, newWidth, newHeight);
            result = result(cropRect).clone();
        }
    }
    
    return removeRemainingEdgeArtifactsLight(result, 50);
}


vector<PieceInfo> PieceExtractor::extractPiecesWithInfo(const cv::Mat& rgbImage, const std::string& outputDir) {
    vector<PieceInfo> pieces;

    if (rgbImage.empty()) {
        cerr << "Empty image passed to extractPiecesWithInfo" << endl;
        return pieces;
    }

    bool savePieces = !outputDir.empty();
    if (savePieces) {
        filesystem::create_directories(outputDir);
    }

    cv::Mat gray;
    cv::cvtColor(rgbImage, gray, cv::COLOR_RGB2GRAY);

    // Threshold
    cv::Mat mask;
    cv::threshold(gray, mask, 10, 255, cv::THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int pieceIndex = 0;
    const double MIN_AREA = 100;

    for (const auto& contour : contours) {
        if (contourArea(contour) < MIN_AREA) continue;

        cv::RotatedRect box = cv::minAreaRect(contour);
        float angle = box.angle;
        cv::Size2f boxSize = box.size;
        
        normalizeRotatedRect(angle, boxSize);

        if (boxSize.width < 10 || boxSize.height < 10) continue;

        Mat pieceImg;

        if (abs(angle) > 1.0f) {
            Mat M = getRotationMatrix2D(box.center, angle, 1.0);
            Mat rotated;
            warpAffine(rgbImage, rotated, M, rgbImage.size(), INTER_LANCZOS4, BORDER_REFLECT);

            cv::Rect roi;
            roi.width  = (int)boxSize.width;
            roi.height = (int)boxSize.height;
            roi.x = (int)(box.center.x - roi.width  / 2);
            roi.y = (int)(box.center.y - roi.height / 2);

            roi.x = max(0, min(roi.x, rotated.cols - 1));
            roi.y = max(0, min(roi.y, rotated.rows - 1));
            if (roi.x + roi.width > rotated.cols)  roi.width  = rotated.cols - roi.x;
            if (roi.y + roi.height > rotated.rows) roi.height = rotated.rows - roi.y;

            if (roi.width > 0 && roi.height > 0) {
                // Create padded ROI
                Rect roi_padded = roi;
                roi_padded.x = max(0, roi.x - 1);
                roi_padded.y = max(0, roi.y - 1);
                roi_padded.width = min(rotated.cols - roi_padded.x, roi.width + 2);
                roi_padded.height = min(rotated.rows - roi_padded.y, roi.height + 2);
                
                Mat piece_with_padding = rotated(roi_padded).clone();
                Rect inner_roi(1, 1, roi.width, roi.height);
                pieceImg = piece_with_padding(inner_roi).clone();
            }
        } else {
            Rect bbox = boundingRect(contour);
            bbox.x = max(0, bbox.x);
            bbox.y = max(0, bbox.y);
            bbox.width = min(rgbImage.cols - bbox.x, bbox.width);
            bbox.height = min(rgbImage.rows - bbox.y, bbox.height);
            if (bbox.width <= 0 || bbox.height <= 0) continue;
            
            pieceImg = rgbImage(bbox).clone();
        }


        Scalar mean = cv::mean(pieceImg);
        if (mean[0] + mean[1] + mean[2] < 30) continue;

        Mat cleanedPiece = cleanExtractionArtifacts(pieceImg, 15);

        PieceInfo info;
        info.img = cleanedPiece;
        // info.img = pieceImg;
        info.center = box.center;
        info.originalRotation = angle;
        info.originalSize = Size(static_cast<int>(boxSize.width), static_cast<int>(boxSize.height));
        pieces.push_back(info);

        if (savePieces) {
            imwrite(outputDir + "/piece_" + to_string(pieceIndex) + ".png", cleanedPiece);
        }
        pieceIndex++;
    }

    return pieces;
}
// vector<PieceInfo> PieceExtractor::extractPiecesWithInfo(const Mat& bgrImage, const string& outputDir) {
//     vector<PieceInfo> pieces;

//     if (bgrImage.empty()) {
//         cerr << "Empty image passed to extractPiecesWithInfo" << endl;
//         return pieces;
//     }

//     bool savePieces = !outputDir.empty();
//     if (savePieces) {
//         filesystem::create_directories(outputDir);
//     }

//     Mat gray, mask;
//     cvtColor(bgrImage, gray, COLOR_BGR2GRAY);
//     threshold(gray, mask, 15, 255, THRESH_BINARY);

//     Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//     morphologyEx(mask, mask, MORPH_CLOSE, kernel);
//     morphologyEx(mask, mask, MORPH_OPEN, kernel);

//     vector<vector<Point>> contours;
//     findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//     sortContoursByArea(contours);

//     int pieceIndex = 0;
//     const double MIN_AREA = 100;

//     for (const auto& contour : contours) {
//         if (contourArea(contour) < MIN_AREA) continue;

//         RotatedRect box = minAreaRect(contour);
//         float angle = box.angle;
//         Size2f boxSize = box.size;
//         normalizeRotatedRect(angle, boxSize);

//         if (boxSize.width < 10 || boxSize.height < 10) continue;

//         Mat pieceImg;

//         if (abs(angle) > 1.0f) {
//             Mat rotationMatrix = getRotationMatrix2D(box.center, angle, 1.0);
//             Mat rotated;
//             warpAffine(bgrImage, rotated, rotationMatrix, bgrImage.size(), INTER_CUBIC, BORDER_REPLICATE);

//             Rect roi = computeSafeROI(box.center, boxSize, rotated.size());
//             if (roi.width <= 0 || roi.height <= 0) continue;
//             pieceImg = rotated(roi).clone();
//         } else {
//             Rect bbox = box.boundingRect() & Rect(0, 0, bgrImage.cols, bgrImage.rows);
//             if (bbox.width <= 0 || bbox.height <= 0) continue;
//             pieceImg = bgrImage(bbox).clone();
//         }

//         Scalar mean = cv::mean(pieceImg);
//         if (mean[0] + mean[1] + mean[2] < 30) continue;

//         PieceInfo info;
//         info.img = pieceImg;
//         info.center = box.center;
//         info.originalRotation = angle;
//         info.originalSize = Size(static_cast<int>(boxSize.width), static_cast<int>(boxSize.height));
//         pieces.push_back(info);

//         if (savePieces) {
//             imwrite(outputDir + "/piece_" + to_string(pieceIndex) + ".png", pieceImg);
//         }
//         pieceIndex++;
//     }

//     return pieces;
// }

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