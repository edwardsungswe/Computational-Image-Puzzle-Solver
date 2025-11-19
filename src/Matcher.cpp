#include "Matcher.h"
#include <opencv2/opencv.hpp>
#include <limits>

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
                }
            }
        }
        sort(matches.begin(), matches.end(),
            [](const Pair& a, const Pair& b) {
                return a.val < b.val;
            });
    }
    return matches;
}

// PuzzleLayout buildLayout(int n)
// {
//     int s = sqrt(n);
//     PuzzleLayout pl;
//     pl.grid.resize(s, vector<int>(s, -1));

//     int k = 0;
//     for (int r = 0; r < s; r++)
//         for (int c = 0; c < s; c++)
//             pl.grid[r][c] = k++;

//     return pl;
// }

PuzzleLayout buildLayout(const vector<Pair>& matches, const vector<PieceFeature>& f)
{
    PuzzleLayout layout;
    int s = f.size();

    unordered_map<int, PiecePosition> position;
    unordered_map<int, bool> placed;

    Pair match = matches[0];
    cv::Size sizeA = f[match.pieceA].img.size();
    cv::Size sizeB = f[match.pieceB].img.size();

    cv::Point2f newPosition = findCoords(sizeA, sizeB, );

    position[match.pieceA] = {cv::Point2f(0, 0), 0, sizeA};
    position[match.pieceB] = {newPosition, 0, sizeB}

    placed[match.pieceA] = true;
    placed[match.pieceB] = true;

}

}
