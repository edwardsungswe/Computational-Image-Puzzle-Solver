#include "Matcher.h"
#include <opencv2/opencv.hpp>
#include <limits>
#include <queue>
#include <unordered_map>
#include <algorithm>

using namespace std;
using namespace cv;


static double distLuma(const EdgeFeature& a, const EdgeFeature& b)
{
    double s = 0;
    for (int i = 0; i < a.vals.size(); i++) {
        double d = a.vals[i] - b.vals[i];
        s += d * d;
    }
    return s;
}


static double distColor(const Mat& a, const Mat& b)
{
    double s = 0;
    for (int i = 0; i < a.rows; i++) {
        Vec3b pa = a.at<Vec3b>(i, 0);
        Vec3b pb = b.at<Vec3b>(i, 0);

        double dh = pa[0] - pb[0];
        double ds = pa[1] - pb[1];
        double dv = pa[2] - pb[2];

        s += dh*dh + ds*ds + dv*dv;
    }
    return s;
}

static double distGradient(const Mat& a, const Mat& b)
{
    double s = 0;
    for (int i = 0; i < a.rows; i++) {

        double da = a.at<uchar>(i, 0);
        double db = b.at<uchar>(i, 0);

        double d = da - db;
        s += d * d;
    }
    return s;
}

Mat matEdgeHSV(const Mat& img, int edge, int N)
{
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    int h = hsv.rows, w = hsv.cols;

    Mat out(N, 1, CV_8UC3);

    for (int i = 0; i < N; i++) {
        float t = float(i) / (N - 1);
        int x, y;

        switch(edge){
            case 0: x = t*(w-1); y = 0;       break;
            case 1: x = w-1;     y = t*(h-1); break;
            case 2: x = t*(w-1); y = h-1;     break;
            default: x = 0;      y = t*(h-1); break;
        }

        out.at<Vec3b>(i,0) = hsv.at<Vec3b>(y,x);
    }
    return out;
}

Mat matEdgeGrad(const Mat& imgGray, int edge, int N)
{
    int h = imgGray.rows, w = imgGray.cols;
    Mat grad(N, 1, CV_8UC1);

    Mat gx, gy;
    Sobel(imgGray, gx, CV_32F, 1,0);
    Sobel(imgGray, gy, CV_32F, 0,1);

    Mat mag;
    magnitude(gx, gy, mag);

    for (int i = 0; i < N; i++) {
        float t = float(i)/(N-1);
        int x, y;

        switch(edge){
            case 0: x = t*(w-1); y = 0; break;
            case 1: x = w-1;     y = t*(h-1); break;
            case 2: x = t*(w-1); y = h-1; break;
            default: x = 0;      y = t*(h-1); break;
        }

        grad.at<uchar>(i,0) = mag.at<float>(y,x);
    }
    return grad;
}

double edgeDistanceFull(const PieceFeature& A, const PieceFeature& B, int edgeA, int edgeB)
{
    const int N = 40;

    Mat Ahsv = matEdgeHSV(A.img, edgeA, N); // updated to any edge on A
    Mat Bhsv = matEdgeHSV(B.img, edgeB, N); // updated to any edge on B

    Mat Agray, Bgray;
    cvtColor(A.img, Agray, COLOR_BGR2GRAY);
    cvtColor(B.img, Bgray, COLOR_BGR2GRAY);

    Mat Agrad = matEdgeGrad(Agray, edgeA, N);
    Mat Bgrad = matEdgeGrad(Bgray, edgeB, N);

    const EdgeFeature& efA = (edgeA == 0) ? A.top : (edgeA == 1) ? A.right : (edgeA == 2) ? A.bottom : A.left;
    const EdgeFeature& efB = (edgeB == 0) ? B.top : (edgeB == 1) ? B.right : (edgeB == 2) ? B.bottom : B.left;

    double L = distLuma(efA, efB);
    double C = distColor(Ahsv, Bhsv);
    double G = distGradient(Agrad, Bgrad);

    double dist = 
        1.0 * L +
        0.5 * C +
        0.8 * G;

    return dist;
}

namespace Matcher {

vector<Pair> matchAll(const vector<PieceFeature>& f)
{
    vector<Pair> matches;
    int N = f.size();
    vector<pair<int,double>> res(N);

    for (int i = 0; i < N; i++) {
        double best = 1e18;
        int bestIdx = -1;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            for (int edgeA = 0; edgeA < 4; edgeA++) {
                for (int edgeB = 0; edgeB < 4; edgeB++) {
                    double d = edgeDistanceFull(f[i], f[j], edgeA, edgeB);

                    Pair match = {i, j, edgeA, edgeB, d};
                    matches.push_back({i, j, edgeA, edgeB, d});
                }
            }
        }
    }
    sort(matches.begin(), matches.end(),
    [](const Pair& a, const Pair& b) {
        return a.val < b.val;
    });
    return matches;
}

PiecePosition calculatePlacement(int currSide, const cv::Size& currSize, int otherSide, const cv::Size& otherSize) 
{
    const float rightX = currSize.width;
    const float leftX = -otherSize.width;
    const float downY = currSize.height;
    const float upY = -otherSize.height;
    
    struct Placement { float x, y, rot; };
    Placement result = {rightX, 0, 0};
    
    if (currSide == 1) { // RIGHT
        if (otherSide == 3) result = {rightX, 0, 0};
        else if (otherSide == 0) result = {rightX, upY, 90};
        else if (otherSide == 2) result = {rightX, downY, 270};
        else if (otherSide == 1) result = {rightX + otherSize.width, 0, 180};
    }
    else if (currSide == 3) { // LEFT
        if (otherSide == 1) result = {leftX, 0, 0};
        else if (otherSide == 0) result = {leftX, upY, 270};
        else if (otherSide == 2) result = {leftX, downY, 90};
        else if (otherSide == 3) result = {leftX - currSize.width, 0, 180};
    }
    else if (currSide == 2) { // BOTTOM
        if (otherSide == 0) result = {0, downY, 0};
        else if (otherSide == 1) result = {otherSize.width, downY, 270};
        else if (otherSide == 3) result = {leftX, downY, 90};
        else if (otherSide == 2) result = {0, downY + otherSize.height, 180};
    }
    else if (currSide == 0) { // TOP
        if (otherSide == 2) result = {0, upY, 0};
        else if (otherSide == 1) result = {otherSize.width, upY, 90};
        else if (otherSide == 3) result = {leftX, upY, 270};
        else if (otherSide == 0) result = {0, upY - currSize.height, 180};
    }
    
    return {cv::Point2f(result.x, result.y), result.rot, otherSize};
}

cv::Rect2f findTotalArea(const unordered_map<int, PiecePosition>& locations) 
{
    if (locations.empty()) {
        return cv::Rect2f(0, 0, 0, 0);
    }
    
    float left = 1e9, top = 1e9, right = -1e9, bottom = -1e9;
    
    for (const auto& entry : locations) {
        const PiecePosition& spot = entry.second;
        float pieceLeft = spot.position.x;
        float pieceTop = spot.position.y;
        float pieceRight = pieceLeft + spot.size.width;
        float pieceBottom = pieceTop + spot.size.height;
        
        if (pieceLeft < left) left = pieceLeft;
        if (pieceTop < top) top = pieceTop;
        if (pieceRight > right) right = pieceRight;
        if (pieceBottom > bottom) bottom = pieceBottom;
    }
    
    return cv::Rect2f(left, top, right - left, bottom - top);
}

PuzzleLayout buildLayout(const vector<Pair>& matches, const vector<PieceFeature>& f)
{
    PuzzleLayout layout;
    int s = f.size();
    if (s == 0) return layout;
    
    queue<int> q;
    unordered_map<int, PiecePosition> positions;
    unordered_map<int, bool> placed;

    if (matches.empty()) {
        for (int i = 0; i < s; i++) {
            positions[i] = {cv::Point2f((i % 3) * 150, (i / 3) * 150), 0, f[i].img.size()};
        }
        layout.position = positions;
        layout.bounds = findTotalArea(positions);
        return layout;
    }

    Pair match = matches[0];
    cv::Size sizeA = f[match.pieceA].img.size();
    cv::Size sizeB = f[match.pieceB].img.size();

    PiecePosition placement = calculatePlacement(match.edgeA, sizeA, match.edgeB, sizeB);

    positions[match.pieceA] = {cv::Point2f(0, 0), 0, sizeA};
    positions[match.pieceB] = {placement.position, placement.rotation, sizeB};

    placed[match.pieceA] = true;
    placed[match.pieceB] = true;

    q.push(match.pieceA);
    q.push(match.pieceB);

    while (placed.size() < s && !q.empty()) {
        int currentPiece = q.front();
        cv::Point2f currentPosition = positions[currentPiece].position;
        cv::Size currentSize = positions[currentPiece].size;
        q.pop();

        for (const auto& nextMatch : matches) {
            if (nextMatch.pieceA != currentPiece && nextMatch.pieceB != currentPiece) continue;

            int neighbor = (nextMatch.pieceA == currentPiece) ? nextMatch.pieceB : nextMatch.pieceA;
            if (placed[neighbor]) continue;

            int currentEdge, otherEdge;
            if (nextMatch.pieceA == currentPiece) {
                currentEdge = nextMatch.edgeA;
                otherEdge = nextMatch.edgeB;
            } else {
                currentEdge = nextMatch.edgeB;
                otherEdge = nextMatch.edgeA;
            }

            cv::Size neighborSize = f[neighbor].img.size();
            PiecePosition neighborPlacement = calculatePlacement(currentEdge, currentSize, otherEdge, neighborSize);
            cv::Point2f newPosition = currentPosition + neighborPlacement.position;

            positions[neighbor] = {newPosition, neighborPlacement.rotation, neighborSize};
            placed[neighbor] = true;
            q.push(neighbor);
        }
    }

    for (int i = 0; i < s; i++) {
        if (!placed[i]) {
            positions[i] = {cv::Point2f(rand() % 500, rand() % 500), 0, f[i].img.size()};
        }
    }

    layout.position = positions;
    layout.bounds = findTotalArea(positions);
    return layout;
}


}
