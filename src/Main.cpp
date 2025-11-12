#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

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

    unsigned char* inData = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));
    for (int i = 0; i < width * height; i++) {
        inData[3 * i]     = Rbuf[i];
        inData[3 * i + 1] = Gbuf[i];
        inData[3 * i + 2] = Bbuf[i];
    }

    return inData;
}

int main() {
    string path = "C:/Users/7dann/Documents/CS/CSCI-576-starter-code/data_sample/data_sample/starry_night_rotate.rgb";

    string pathPNG = "C:/Users/7dann/Documents/CS/CSCI-576-starter-code/data_sample/data_sample/starry_night_rotate.png";;

    // Load the image
    Mat img = imread(pathPNG, IMREAD_COLOR);


    // Get width and height
    int width  = img.cols;
    int height = img.rows;

    // Read the raw RGB file
    unsigned char* buffer = readImageData(path, width, height);

    // Create OpenCV Mat from buffer (interleaved RGB)
    Mat rgbImage(height, width, CV_8UC3, buffer);

    // Convert RGB -> YUV
    Mat yuvImage;
    cvtColor(rgbImage, yuvImage, COLOR_RGB2YUV);

    // Create grayscale mask (to detect non-black pixels)
    Mat gray;
    cvtColor(rgbImage, gray, COLOR_RGB2GRAY);

    Mat mask;
    threshold(gray, mask, 10, 255, THRESH_BINARY);

    // Find contours around each piece
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contourImage = rgbImage.clone();
    vector<Mat> pieces;
    int idx = 0;

    for (const auto& c : contours) {
        RotatedRect box = minAreaRect(c);

        Point2f ptsf[4];
        box.points(ptsf);
        vector<Point> poly;
        for (int i = 0; i < 4; ++i) {
            poly.emplace_back(cv::Point(cvRound(ptsf[i].x), cvRound(ptsf[i].y)));
        }
        polylines(contourImage, poly, true, Scalar(0,0,255), 2);

        float angle = box.angle;
        Size2f boxSize = box.size;
        if (box.angle < -45.0f) {
            angle += 90.0f;
            swap(boxSize.width, boxSize.height);
        }

        Mat M = getRotationMatrix2D(box.center, angle, 1.0);

        Mat rotated;
        warpAffine(rgbImage, rotated, M, rgbImage.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0,0,0));

        Rect roi;
        roi.width  = static_cast<int>(boxSize.width);
        roi.height = static_cast<int>(boxSize.height);
        roi.x = static_cast<int>(box.center.x - roi.width / 2.0f);
        roi.y = static_cast<int>(box.center.y - roi.height / 2.0f);

        // clamp ROI to image bounds
        roi.x = max(0, min(roi.x, rotated.cols - 1));
        roi.y = max(0, min(roi.y, rotated.rows - 1));
        if (roi.x + roi.width > rotated.cols)  roi.width  = rotated.cols - roi.x;
        if (roi.y + roi.height > rotated.rows) roi.height = rotated.rows - roi.y;

        if (roi.width <= 0 || roi.height <= 0) continue;

        Mat piece = rotated(roi).clone();
        pieces.push_back(piece);

    }


    cout << "Detected " << pieces.size() << " pieces." << endl;


    free(buffer);

    imshow("Detected Pieces", contourImage);

    cout << "Press any key in the image windows to exit..." << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}
