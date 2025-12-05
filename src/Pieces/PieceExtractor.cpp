#include "PieceExtractor.h"
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>
using namespace std;
using namespace cv;

std::vector<cv::Mat> PieceExtractor::extractPieces(const cv::Mat& rgbImage, bool enableRotation, 
                                                    const std::string& outputDir) {
    std::vector<cv::Mat> pieces;

    // if outputDir is not empty, create directory. Debug 
    bool savePieces = !outputDir.empty();
    // if (savePieces) {
    //     try {
    //         std::filesystem::create_directories(outputDir);
    //         std::cout << "Saving extracted pieces to: " << outputDir << std::endl;
    //     } catch (const std::exception& e) {
    //         std::cerr << "Warning: Failed to create output directory '" << outputDir 
    //                   << "': " << e.what() << std::endl;
    //         savePieces = false;
    //     }
    // }

    // Convert RGB → Gray
    cv::Mat gray;
    cv::cvtColor(rgbImage, gray, cv::COLOR_RGB2GRAY);

    // Threshold
    cv::Mat mask;
    cv::threshold(gray, mask, 10, 255, cv::THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // For each contour, extract rotated piece
    int pieceIndex = 0;
    for (const auto& c : contours) {
        if (!enableRotation) {
            cv::Rect box = cv::boundingRect(c);
            box.x = max(0, box.x);
            box.y = max(0, box.y);
            box.width = min(rgbImage.cols - box.x, box.width);
            box.height = min(rgbImage.rows - box.y, box.height);
            if (box.width <= 0 || box.height <= 0)
                continue;
            cv::Mat piece = rgbImage(box).clone();
            pieces.push_back(piece);
            
            // //Debug: save pieces to directory
            // if (savePieces) {
            //     std::filesystem::path filePath = std::filesystem::path(outputDir);
            //     std::ostringstream filename;
            //     filename << "piece_" << std::setfill('0') << std::setw(4) 
            //              << pieceIndex << ".png";
            //     filePath /= filename.str();
            //     cv::imwrite(filePath.string(), piece);
            //     std::cout << "Saved: " << filePath.string() 
            //               << " (size: " << piece.cols << "x" << piece.rows << ")" << std::endl;
            // }
            pieceIndex++;
            continue;

        } else {
            cv::RotatedRect box = cv::minAreaRect(c);
            float angle = box.angle;
            cv::Size2f boxSize = box.size;

            if (box.angle < -45.0f) {
                angle += 90.0f;
                std::swap(boxSize.width, boxSize.height);
            }

            cv::Mat M = cv::getRotationMatrix2D(box.center, angle, 1.0);

            cv::Mat rotated;
            cv::warpAffine(rgbImage, rotated, M, rgbImage.size(),
                        cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

            cv::Rect roi;
            roi.width  = static_cast<int>(boxSize.width);
            roi.height = static_cast<int>(boxSize.height);
            roi.x = static_cast<int>(box.center.x - roi.width / 2.0f);
            roi.y = static_cast<int>(box.center.y - roi.height / 2.0f);

            // Clamp ROI
            roi.x = std::max(0, std::min(roi.x, rotated.cols - 1));
            roi.y = std::max(0, std::min(roi.y, rotated.rows - 1));
            if (roi.x + roi.width > rotated.cols)  roi.width  = rotated.cols - roi.x;
            if (roi.y + roi.height > rotated.rows) roi.height = rotated.rows - roi.y;

            if (roi.width <= 0 || roi.height <= 0)
                continue;

            cv::Mat piece = rotated(roi).clone();
            pieces.push_back(piece);
            
            // Debug: save pieces to directory
            // if (savePieces) {
            //     std::filesystem::path filePath = std::filesystem::path(outputDir);
            //     std::ostringstream filename;
            //     filename << "piece_" << std::setfill('0') << std::setw(4) 
            //              << pieceIndex << ".png";
            //     filePath /= filename.str();
            //     cv::imwrite(filePath.string(), piece);
            //     std::cout << "Saved: " << filePath.string() 
            //               << " (size: " << piece.cols << "x" << piece.rows << ")" << std::endl;
            // }
            pieceIndex++;
        }
    }
    
    if (savePieces) {
        std::cout << "Total pieces extracted and saved: " << pieces.size() << std::endl;
    }
    
    return pieces;
}

cv::Mat PieceExtractor::drawContoursOnImage(const cv::Mat& rgbImage) {
    cv::Mat gray;
    cv::cvtColor(rgbImage, gray, cv::COLOR_RGB2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 10, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat contourImage = rgbImage.clone();
    for (auto& c : contours) {
        cv::RotatedRect box = cv::minAreaRect(c);

        cv::Point2f pts[4];
        box.points(pts);

        std::vector<cv::Point> poly;
        for (int i = 0; i < 4; i++)
            poly.emplace_back(cv::Point((int)pts[i].x, (int)pts[i].y));

        cv::polylines(contourImage, poly, true, cv::Scalar(0,0,255), 2);
    }

    return contourImage;
}
