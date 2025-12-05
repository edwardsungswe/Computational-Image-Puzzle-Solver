#include "PieceMatcher.h"
#include <opencv2/opencv.hpp>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <deque>
#include <queue>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>


using namespace std;
using namespace cv;
// static std::ofstream scoreLog;
// struct ScoreBreakdown {
//     double L_norm = 0.0;
//     double D_ncc  = 0.0;
//     double G_norm = 0.0;
//     double P_norm = 0.0;
//     double T_norm = 0.0;
//     double colorNorm = 0.0;
//     double combined = 0.0;
//     bool nccFail = false;
//     bool colorFail = false;
// };

// void saveMatchesToFile(const vector<Pair>& matches, const string& filename) {
//     ofstream out(filename);
//     if (!out.is_open()) {
//         cerr << "Failed to open file: " << filename << endl;
//         return;
//     }

//     out << "PieceA\tEdgeA\tPieceB\tEdgeB\tScore\n";
//     out << fixed << setprecision(4);

//     for (const auto& m : matches) {
//         out << m.pieceA << "\t" << m.edgeA << "\t"
//             << m.pieceB << "\t" << m.edgeB << "\t"
//             << m.val << "\n";
//     }

//     out.close();
// }

// static void initScoreLog(const std::string& path="C:\\Users\\7dann\\Documents\\CS\\edge_score_log.csv") {
//     scoreLog.open(path, std::ios::out | std::ios::trunc);
//     if (!scoreLog.is_open()) {
//         cerr << "initScoreLog: FAILED to open '" << path << "'\n";
//     } else {
//         cout << "initScoreLog: open -> '" << path << "'\n";
//         scoreLog << "pieceA,edgeA,pieceB,edgeB,L_norm,D_ncc,G_norm,P_norm,T_norm,colorNorm,nccFail,colorFail,combined\n";
//         scoreLog.flush();
//     }
// }


static double distLuma(const EdgeFeature& a, const EdgeFeature& b)
{
    double s = 0;
    // deal with different length of feature vectors
    int minSize = min(a.vals.size(), b.vals.size());
    for (int i = 0; i < minSize; i++) {
        double d = a.vals[i] - b.vals[i];
        s += d * d;
    }
    // if length is different, penalize the extra part
    if (a.vals.size() != b.vals.size()) {
        s += abs(static_cast<int>(a.vals.size() - b.vals.size())) * 100.0;
    }
    return s;
}

// RGB sample point distance (preserve spatial order, for shape matching)
static double distRGBProfile(const EdgeFeature& a, const EdgeFeature& b)
{
    if (a.rgbVals.empty() || b.rgbVals.empty()) {
        return distLuma(a, b);  // fallback to luma
    }
    
    double s = 0.0;
    int minSize = min(a.rgbVals.size(), b.rgbVals.size());
    for (int i = 0; i < minSize; i++) {
        Vec3d diff = a.rgbVals[i] - b.rgbVals[i];
        s += diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
    }
    if (a.rgbVals.size() != b.rgbVals.size()) {
        s += abs(static_cast<int>(a.rgbVals.size() - b.rgbVals.size())) * 100.0;
    }
    return s;
}

// RGB histogram distance
static double distHistogram(const EdgeFeature& a, const EdgeFeature& b)
{
    if (a.histogram.empty() || b.histogram.empty()) {
        return 0.0;  // if no histogram, return 0
    }
    
    // L2 distance
    double s = 0.0;
    int minSize = min(a.histogram.size(), b.histogram.size());
    for (int i = 0; i < minSize; i++) {
        double d = a.histogram[i] - b.histogram[i];
        s += d * d;
    }
    if (a.histogram.size() != b.histogram.size()) {
        s += abs(static_cast<int>(a.histogram.size() - b.histogram.size())) * 0.1;
    }
    return s;
}

// chi-square distance
static double distHistogramChiSquare(const EdgeFeature& a, const EdgeFeature& b)
{
    if (a.histogram.empty() || b.histogram.empty()) {
        return distHistogram(a, b);
    }
    
    double s = 0.0;
    int minSize = min(a.histogram.size(), b.histogram.size());
    for (int i = 0; i < minSize; i++) {
        double sum = a.histogram[i] + b.histogram[i];
        if (sum > 1e-6) {
            double diff = a.histogram[i] - b.histogram[i];
            s += (diff * diff) / sum;
        }
    }
    return s * 0.5; 
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

double computeEdgeProfileCompatibility(const EdgeFeature& efA, const EdgeFeature& efB) {
    // use RGB sample points first, if not, use grayscale values
    bool useRGB = !efA.rgbVals.empty() && !efB.rgbVals.empty();
    
    if (useRGB) {
        // use RGB sample points for shape matching
        vector<Vec3d> reversedB = efB.rgbVals;
        reverse(reversedB.begin(), reversedB.end());
        
        double forwardMatch = 0.0, reverseMatch = 0.0;
        double forwardSlopeConsistency = 0.0, reverseSlopeConsistency = 0.0;
        
        int minSize = min(efA.rgbVals.size(), efB.rgbVals.size());
        for (int i = 0; i < minSize; i++) {
            Vec3d diffForward = efA.rgbVals[i] - efB.rgbVals[i];
            Vec3d diffReverse = efA.rgbVals[i] - reversedB[i];
            forwardMatch += diffForward[0]*diffForward[0] + diffForward[1]*diffForward[1] + diffForward[2]*diffForward[2];
            reverseMatch += diffReverse[0]*diffReverse[0] + diffReverse[1]*diffReverse[1] + diffReverse[2]*diffReverse[2];
            
            if (i > 0) {
                Vec3d slopeA = efA.rgbVals[i] - efA.rgbVals[i-1];
                Vec3d slopeB_forward = efB.rgbVals[i] - efB.rgbVals[i-1];
                Vec3d slopeB_reverse = reversedB[i] - reversedB[i-1];
                
                Vec3d slopeDiffForward = slopeA + slopeB_forward;
                Vec3d slopeDiffReverse = slopeA + slopeB_reverse;
                forwardSlopeConsistency += sqrt(slopeDiffForward[0]*slopeDiffForward[0] + 
                                                slopeDiffForward[1]*slopeDiffForward[1] + 
                                                slopeDiffForward[2]*slopeDiffForward[2]);
                reverseSlopeConsistency += sqrt(slopeDiffReverse[0]*slopeDiffReverse[0] + 
                                               slopeDiffReverse[1]*slopeDiffReverse[1] + 
                                               slopeDiffReverse[2]*slopeDiffReverse[2]);
            }
        }
        
        double profileScore = min(forwardMatch, reverseMatch);
        double slopeScore = min(forwardSlopeConsistency, reverseSlopeConsistency);
        return profileScore + slopeScore * 0.1;
    } else {
        // fallback to grayscale values
        vector<double> reversedB = efB.vals;
        reverse(reversedB.begin(), reversedB.end());
        
        double forwardMatch = 0.0, reverseMatch = 0.0;
        double forwardSlopeConsistency = 0.0, reverseSlopeConsistency = 0.0;
        
        int minSize = min(efA.vals.size(), efB.vals.size());
        for (int i = 0; i < minSize; i++) {
            double diffForward = efA.vals[i] - efB.vals[i];
            double diffReverse = efA.vals[i] - reversedB[i];
            forwardMatch += diffForward * diffForward;
            reverseMatch += diffReverse * diffReverse;
            
            if (i > 0) {
                double slopeA = efA.vals[i] - efA.vals[i-1];
                double slopeB_forward = efB.vals[i] - efB.vals[i-1];
                double slopeB_reverse = reversedB[i] - reversedB[i-1];
                
                forwardSlopeConsistency += abs(slopeA + slopeB_forward);
                reverseSlopeConsistency += abs(slopeA + slopeB_reverse);
            }
        }

        double profileScore = min(forwardMatch, reverseMatch);
        double slopeScore = min(forwardSlopeConsistency, reverseSlopeConsistency);
        return profileScore + slopeScore * 0.1;
    }
}

Mat extractTexturePatch(const Mat& img, int edge, int distanceFromEdge, int patchWidth) {
    int h = img.rows, w = img.cols;
    Mat patch(patchWidth, 1, CV_8UC3);
    
    for (int i = 0; i < patchWidth; i++) {
        float t = float(i) / (patchWidth - 1);
        int x, y;
        
        switch(edge) {
            case 0:
                x = t * (w-1);
                y = distanceFromEdge;
                break;
            case 1:
                x = w - 1 - distanceFromEdge;
                y = t * (h-1);
                break;
            case 2:
                x = t * (w-1);
                y = h - 1 - distanceFromEdge;
                break;
            case 3:
                x = distanceFromEdge;
                y = t * (h-1);
                break;
        }
        
        x = max(0, min(x, w-1));
        y = max(0, min(y, h-1));
        patch.at<Vec3b>(i, 0) = img.at<Vec3b>(y, x);
    }
    return patch;
}


double computeTextureConsistency(const Mat& imgA, const Mat& imgB, int edgeA, int edgeB) {
    Mat textureA1 = extractTexturePatch(imgA, edgeA, 2, 15);
    Mat textureA2 = extractTexturePatch(imgA, edgeA, 8, 15);
    Mat textureB1 = extractTexturePatch(imgB, edgeB, 2, 15);
    Mat textureB2 = extractTexturePatch(imgB, edgeB, 8, 15);
    
    Mat grayA1, grayA2, grayB1, grayB2;
    cvtColor(textureA1, grayA1, COLOR_BGR2GRAY);
    cvtColor(textureA2, grayA2, COLOR_BGR2GRAY);
    cvtColor(textureB1, grayB1, COLOR_BGR2GRAY);
    cvtColor(textureB2, grayB2, COLOR_BGR2GRAY);
    
    Scalar meanA1, stddevA1, meanA2, stddevA2;
    Scalar meanB1, stddevB1, meanB2, stddevB2;
    meanStdDev(grayA1, meanA1, stddevA1);
    meanStdDev(grayA2, meanA2, stddevA2);
    meanStdDev(grayB1, meanB1, stddevB1);
    meanStdDev(grayB2, meanB2, stddevB2);
    
    double varianceDiff1 = abs(stddevA1[0] - stddevB1[0]);
    double varianceDiff2 = abs(stddevA2[0] - stddevB2[0]);
    double meanEvolutionA = abs(meanA1[0] - meanA2[0]);
    double meanEvolutionB = abs(meanB1[0] - meanB2[0]);
    double evolutionConsistency = abs(meanEvolutionA - meanEvolutionB);
    
    return varianceDiff1 + varianceDiff2 + evolutionConsistency * 2.0;
}


Mat reverseMat(const Mat& m) {
    Mat reversed = m.clone();
    for (int i = 0; i < m.rows / 2; i++) {
        int j = m.rows - 1 - i;
        if (m.channels() == 1) {
            std::swap(reversed.at<uchar>(i, 0), reversed.at<uchar>(j, 0));
        } else if (m.channels() == 3) {
            Vec3b temp = reversed.at<Vec3b>(i, 0);
            reversed.at<Vec3b>(i, 0) = reversed.at<Vec3b>(j, 0);
            reversed.at<Vec3b>(j, 0) = temp;
        }
    }
    return reversed;
}

double edgeDistanceFull(const PieceFeature& A, const PieceFeature& B, int edgeA, int edgeB) {
    const int N = 40;

    Mat Ahsv = matEdgeHSV(A.img, edgeA, N);
    Mat Bhsv = matEdgeHSV(B.img, edgeB, N);

    Mat Agray, Bgray;
    cvtColor(A.img, Agray, COLOR_BGR2GRAY);
    cvtColor(B.img, Bgray, COLOR_BGR2GRAY);

    Mat Agrad = matEdgeGrad(Agray, edgeA, N);
    Mat Bgrad = matEdgeGrad(Bgray, edgeB, N);

    const EdgeFeature& efA = (edgeA == 0) ? A.top : (edgeA == 1) ? A.right : (edgeA == 2) ? A.bottom : A.left;
    const EdgeFeature& efB = (edgeB == 0) ? B.top : (edgeB == 1) ? B.right : (edgeB == 2) ? B.bottom : B.left;

    bool areComplementary = ((edgeA == 0 && edgeB == 2) || (edgeA == 2 && edgeB == 0) ||
                             (edgeA == 1 && edgeB == 3) || (edgeA == 3 && edgeB == 1));
    
    Mat Bhsv_reversed = areComplementary ? reverseMat(Bhsv) : Bhsv;
    Mat Bgrad_reversed = areComplementary ? reverseMat(Bgrad) : Bgrad;
    
    double C_forward = distColor(Ahsv, Bhsv);
    double C_reverse = distColor(Ahsv, Bhsv_reversed);
    double C = min(C_forward, C_reverse);
    
    double G_forward = distGradient(Agrad, Bgrad);
    double G_reverse = distGradient(Agrad, Bgrad_reversed);
    double G = min(G_forward, G_reverse);

    // mixed feature matching: RGB sample points + histogram
    double H = distHistogram(efA, efB);           // RGB histogram
    double RGB = distRGBProfile(efA, efB);        // RGB sample points (spatial order, for color matching)
    double P = computeEdgeProfileCompatibility(efA, efB);  // shape matching (using RGB sample points)
    double T = computeTextureConsistency(A.img, B.img, edgeA, edgeB);

    // recommended weight configuration:
    // - Profile (P): most important, for matching edge shape (convex/concave)
    // - RGB sample points: for color and texture matching
    // - histogram: auxiliary feature, robust to noise
    // - gradient: edge strength
    // - HSV color: auxiliary color matching
    double dist = 4.0 * P + 2.0 * RGB + 1.0 * H + 0.3 * C + 1.0 * G + 1.5 * T;

 

    return dist;
}

namespace PieceMatcher {
vector<Pair> createFilteredMatches(const vector<PieceFeature>& features, double ratioTestThreshold)
{
    vector<Pair> matches;
    // ofstream debugLog("match_analysis.csv");
    // debugLog << "PieceA,EdgeA,PieceB,EdgeB,Luma,Color,Gradient,Profile,Texture,Continuity,TotalScore,Status\n";
    int pieceCount = features.size();
    
    for (int i = 0; i < pieceCount; i++) {
        for (int j = i + 1; j < pieceCount; j++) {
            for (int edgeA = 0; edgeA < 4; edgeA++) {
                for (int edgeB = 0; edgeB < 4; edgeB++) {
                     if ((edgeA == 0 && edgeB == 2) ||
                        (edgeA == 2 && edgeB == 0) || 
                        (edgeA == 1 && edgeB == 3) || 
                        (edgeA == 3 && edgeB == 1)) { 
                        double score = edgeDistanceFull(features[i], features[j], edgeA, edgeB);
                    
                        matches.push_back({i, j, edgeA, edgeB, score});

                        }
                }
            }
        }
    }
    
    sort(matches.begin(), matches.end(), 
        [](const Pair& a, const Pair& b) { return a.val < b.val; });
    
    // ensure more pieces can be matched
    int checkFirstN = min(30, (int)matches.size());  // increase check count
    double avgTopScores = 0;
    if (checkFirstN > 0) {
        for (int i = 0; i < checkFirstN; i++) {
            avgTopScores += matches[i].val;
        }
        avgTopScores /= checkFirstN;
    } else {
        avgTopScores = 1000.0;  // if no match, set a very large value
    }
    
    // loosen cutoff: from 4.0 to 6.0, allow more matches to pass
    double scoreCutoff = avgTopScores * 6.0;
    
    cout << "Match filtering: avgTopScore=" << avgTopScores 
         << ", cutoff=" << scoreCutoff 
         << ", total matches=" << matches.size() << endl;
    
    vector<Pair> goodMatches;
    unordered_map<int, double> bestScoreForEdge;
    
    for (const auto& match : matches) {
        if (match.val > scoreCutoff) {
            continue;
        }
        
        bool keepMatch = true;
        
        int edgeKeyA = match.pieceA * 10 + match.edgeA;
        if (bestScoreForEdge.count(edgeKeyA)) {
            double ratio = bestScoreForEdge[edgeKeyA] / match.val;
            if (ratio > ratioTestThreshold) {
                keepMatch = false;
            }
        } 
        else {
            bestScoreForEdge[edgeKeyA] = match.val;
        }
        
        int edgeKeyB = match.pieceB * 10 + match.edgeB;
        if (bestScoreForEdge.count(edgeKeyB)) {
            double ratio = bestScoreForEdge[edgeKeyB] / match.val;
            if (ratio > ratioTestThreshold) {
                keepMatch = false;
            }
        } 
        else {
            bestScoreForEdge[edgeKeyB] = match.val;
        }
        
        if (keepMatch) {
            goodMatches.push_back(match);
        }
    }
    
    // saveMatchesToFile(goodMatches, "C:\\Users\\7dann\\Documents\\CS\\matches_ranked.txt");
    
    return goodMatches;
}

cv::Rect2f findTotalArea(const unordered_map<int, PiecePosition>& locations) {
    if (locations.empty()) {
        return cv::Rect2f(0, 0, 0, 0);
    }
    
    float left = 1e9, top = 1e9, right = -1e9, bottom = -1e9;
    
    for (const auto& entry : locations) {
        const PiecePosition& spot = entry.second;
        float pieceLeft = spot.position.x;
        float pieceTop = spot.position.y;
        float pieceRight = pieceLeft + static_cast<float>(spot.size.width);
        float pieceBottom = pieceTop + static_cast<float>(spot.size.height);
        
        if (pieceLeft < left) left = pieceLeft;
        if (pieceTop < top) top = pieceTop;
        if (pieceRight > right) right = pieceRight;
        if (pieceBottom > bottom) bottom = pieceBottom;
    }
    
    return cv::Rect2f(left, top, right - left, bottom - top);
}

Mat rotatePiece(const Mat& img, float rotation) {
    if (rotation == 0.0f) return img.clone();
    
    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat rotationMatrix = getRotationMatrix2D(center, rotation, 1.0);
    
    Mat rotated;
    warpAffine(img, rotated, rotationMatrix, img.size());
    return rotated;
}

// global consistency check: detect left-right swap problem
// check middle column edge matching quality to determine
static double checkGlobalConsistency(const vector<vector<int>>& grid, 
                                     const unordered_map<int, PiecePosition>& positions,
                                     const vector<PieceFeature>& features,
                                     int rows, int cols) {
    if (cols < 2) return 0.0;  // at least 2 columns are needed to check
    
    double totalInconsistency = 0.0;
    int checks = 0;
    
    // check middle column (most likely place for left-right swap)
    int midCol = cols / 2;
    
    for (int r = 0; r < rows; r++) {
        if (grid[r][midCol] >= 0 && midCol > 0 && grid[r][midCol-1] >= 0) {
            int leftPiece = grid[r][midCol-1];
            int rightPiece = grid[r][midCol];
            
            // check match quality of right edge of left piece and left edge of right piece
            double matchScore = edgeDistanceFull(
                features[leftPiece], features[rightPiece],
                1, 3  // right edge vs left edge
            );
            
            // check "flipped" match (if left-right swap)
            // check left edge of left piece and right edge of right piece (this should not match)
            double wrongMatchScore = edgeDistanceFull(
                features[leftPiece], features[rightPiece],
                3, 1  // left edge vs right edge (wrong direction)
            );
            
            // if wrong direction match is better, it means left-right swap may have occurred
            if (wrongMatchScore < matchScore * 0.8) {
                totalInconsistency += (matchScore - wrongMatchScore);
            }
            checks++;
        }
    }
    
    return (checks > 0) ? totalInconsistency / checks : 0.0;
}

float calculateRequiredRotation(int edgeA, int edgeB) {
    if ((edgeA == 0 && edgeB == 2) || (edgeA == 2 && edgeB == 0) ||
        (edgeA == 1 && edgeB == 3) || (edgeA == 3 && edgeB == 1)) {
        return 0.0f; 
    }
    else if ((edgeA == 0 && edgeB == 1) || (edgeA == 1 && edgeB == 2) ||
             (edgeA == 2 && edgeB == 3) || (edgeA == 3 && edgeB == 0)) {
        return 270.0f;
    }
    else if ((edgeA == 0 && edgeB == 3) || (edgeA == 3 && edgeB == 2) ||
             (edgeA == 2 && edgeB == 1) || (edgeA == 1 && edgeB == 0)) {
        return 90.0f;
    }
    else {
        return 180.0f;
    }
}
PuzzleLayout buildLayout(const vector<Pair>& matches, const vector<PieceFeature>& f, int canvasW, int canvasH)
{
    PuzzleLayout layout;
    int N = (int)f.size();

    unordered_map<int, PiecePosition> positions;
    unordered_map<int, vector<bool>> edgeUsed;
    unordered_map<int, int> groupId;
    unordered_map<int, vector<int>> groups;
    int nextGroupId = 0;
    
    for (int i = 0; i < N; ++i) {
        edgeUsed[i] = vector<bool>(4, false);
        groupId[i] = -1;
    }

    const int margin = 50;
    int newGroupXCoordinate = 0;

    cout << "=== STARTING LAYOUT BUILDING ===" << endl;
    // cout << "Total pieces: " << N << ", Total matches: " << matches.size() << endl;

    auto moveGroup = [&](int group, const cv::Point2f& offset) {
        for (int piece : groups[group]) {
            positions[piece].position += offset;
        }
    };

    auto getGroupBounds = [&](int group) -> cv::Rect2f {
        if (groups[group].empty()) return cv::Rect2f(0,0,0,0);
        
        float minX = numeric_limits<float>::max();
        float minY = numeric_limits<float>::max();
        float maxX = numeric_limits<float>::lowest();
        float maxY = numeric_limits<float>::lowest();
        
        for (int piece : groups[group]) {
            cv::Point2f pos = positions[piece].position;
            cv::Size size = positions[piece].size;
            minX = min(minX, pos.x);
            minY = min(minY, pos.y);
            maxX = max(maxX, pos.x + size.width);
            maxY = max(maxY, pos.y + size.height);
        }
        
        return cv::Rect2f(minX, minY, maxX - minX, maxY - minY);
    };

    auto mergeGroups = [&](int groupA, int groupB) {
        if (groupA == groupB) return groupA;
        
        int targetGroup = min(groupA, groupB);
        int sourceGroup = max(groupA, groupB);
        
        for (int piece : groups[sourceGroup]) {
            groups[targetGroup].push_back(piece);
            groupId[piece] = targetGroup;
        }
        groups.erase(sourceGroup);
        return targetGroup;
    };

    auto allPiecesConnected = [&]() {
        if (positions.size() != N) return false;
        

        if (groups.empty()) return false;
        
        int firstGroup = groups.begin()->first;
        for (int i = 0; i < N; i++) {
            if (groupId[i] != firstGroup) {
                return false;
            }
        }
        return true;
    };

    for (size_t i = 0; i < matches.size(); i++) {
        const Pair &m = matches[i];
        int a = m.pieceA;
        int b = m.pieceB;
        int edgeA = m.edgeA;
        int edgeB = m.edgeB;
        
        // cout << "\nProcessing match " << i << " ===" << endl;
        // cout << "Piece " << a << " (edge " << edgeA << ") <-> Piece " << b << " (edge " << edgeB << "), Score: " << m.val << endl;
        
        if (allPiecesConnected()) {
            cout << "All pieces placed, stopping." << endl;
            break;
        }
        
        if (edgeUsed[a][edgeA]) {
            // cout << "SKIP: Edge " << edgeA << " of piece " << a << " already used." << endl;
            continue;
        }
        if (edgeUsed[b][edgeB]) {
            // cout << "SKIP: Edge " << edgeB << " of piece " << b << " already used." << endl;
            continue;
        }

        bool placedA = positions.count(a) != 0;
        bool placedB = positions.count(b) != 0;

        // cout << "Piece " << a << " placed: " << (placedA ? "YES" : "NO") << endl;
        // cout << "Piece " << b << " placed: " << (placedB ? "YES" : "NO") << endl;

        if (!placedA && !placedB) {
            // cout << "CASE: Neither piece placed - creating new group" << endl;
            
            float rotationB = calculateRequiredRotation(edgeA, edgeB);
            cv::Size sizeA = f[a].img.size();
            cv::Size sizeB = f[b].img.size();
            
            // cout << "Piece " << a << " size: " << sizeA.width << "x" << sizeA.height << endl;
            // cout << "Piece " << b << " size: " << sizeB.width << "x" << sizeB.height << endl;
            // cout << "Required rotation for piece " << b << ": " << rotationB << " degrees" << endl;
            

            if (rotationB == 90.0f || rotationB == 270.0f) {
                sizeB = cv::Size(sizeB.height, sizeB.width);
                // cout << "Adjusted piece " << b << " size after rotation: " << sizeB.width << "x" << sizeB.height << endl;
            }

            int newGroup = nextGroupId++;
            groupId[a] = newGroup;
            groupId[b] = newGroup;
            groups[newGroup] = {a, b};

            cv::Point2f posA((float)newGroupXCoordinate, 0.0f);
            
            cv::Point2f offset(0.0f, 0.0f);
            switch (edgeA) {
                case 0: 
                    offset = cv::Point2f(0.0f, - (float)sizeB.height); 
                    // cout << "Edge configuration: A.top <-> B.bottom, placing B above A" << endl;
                    break;
                case 1: 
                    offset = cv::Point2f((float)sizeA.width, 0.0f);
                    // cout << "Edge configuration: A.right <-> B.left, placing B right of A" << endl;
                    break;
                case 2: 
                    offset = cv::Point2f(0.0f, (float)sizeA.height);
                    // cout << "Edge configuration: A.bottom <-> B.top, placing B below A" << endl;
                    break;
                case 3: 
                    offset = cv::Point2f(- (float)sizeB.width, 0.0f);
                    // cout << "Edge configuration: A.left <-> B.right, placing B left of A" << endl;
                    break;
            }
            
            cv::Point2f posB = posA + offset;

            positions[a] = {posA, 0.0f, sizeA};
            positions[b] = {posB, rotationB, sizeB};

            edgeUsed[a][edgeA] = true;
            edgeUsed[b][edgeB] = true;

            // cout << "Placed piece " << a << " at [" << posA.x << ", " << posA.y << "]" << endl;
            // cout << "Placed piece " << b << " at [" << posB.x << ", " << posB.y << "]" << endl;

            cv::Rect2f groupBounds = getGroupBounds(newGroup);
            // cout << "Group " << newGroup << " bounds: [" << groupBounds.x << ", " << groupBounds.y << ", " 
            //      << groupBounds.width << ", " << groupBounds.height << "]" << endl;
                 
            newGroupXCoordinate += (int)groupBounds.width + margin;
            //  cout << "Next group X coordinate: " << newGroupXCoordinate << endl;
            continue;
        }

        if (placedA && !placedB) {
            // cout << "CASE: Piece " << a << " placed, attaching piece " << b << endl;
            
            float rotationB = calculateRequiredRotation(edgeA, edgeB);
            cv::Size sizeA = positions[a].size;
            cv::Size sizeB = f[b].img.size();
            
            // cout << "Piece " << a << " current size: " << sizeA.width << "x" << sizeA.height << endl;
            // cout << "Piece " << b << " original size: " << sizeB.width << "x" << sizeB.height << endl;
            // cout << "Required rotation for piece " << b << ": " << rotationB << " degrees" << endl;
            
            if (rotationB == 90.0f || rotationB == 270.0f) {
                sizeB = cv::Size(sizeB.height, sizeB.width);
                // cout << "Adjusted piece " << b << " size after rotation: " << sizeB.width << "x" << sizeB.height << endl;
            }
            
            cv::Point2f posA = positions[a].position;
            cv::Point2f offset(0.0f, 0.0f);
            switch (edgeA) {
                case 0: 
                    offset = cv::Point2f(0.0f, -sizeB.height); 
                    // cout << "Edge configuration: A.top <-> B.bottom, placing B above A" << endl;
                    break;
                case 1: 
                    offset = cv::Point2f(sizeA.width, 0.0f);
                    // cout << "Edge configuration: A.right <-> B.left, placing B right of A" << endl;
                    break;
                case 2: 
                    offset = cv::Point2f(0.0f, sizeA.height);
                    // cout << "Edge configuration: A.bottom <-> B.top, placing B below A" << endl;
                    break;
                case 3: 
                    offset = cv::Point2f(-sizeB.width, 0.0f);
                    // cout << "Edge configuration: A.left <-> B.right, placing B left of A" << endl;
                    break;
            }
            
            cv::Point2f posB = posA + offset;
            positions[b] = {posB, rotationB, sizeB};

            int groupA = groupId[a];
            groupId[b] = groupA;
            groups[groupA].push_back(b);

            edgeUsed[a][edgeA] = true;
            edgeUsed[b][edgeB] = true;

            // cout << "Attached piece " << b << " at [" << posB.x << ", " << posB.y << "] to group " << groupA << endl;
            continue;
        }

        if (!placedA && placedB) {
            // cout << "CASE: Piece " << b << " placed, attaching piece " << a << endl;
            
            float rotationA = calculateRequiredRotation(edgeA, edgeB);
            cv::Size sizeA = f[a].img.size();
            cv::Size sizeB = positions[b].size;
            
            // cout << "Piece " << a << " original size: " << sizeA.width << "x" << sizeA.height << endl;
            // cout << "Piece " << b << " current size: " << sizeB.width << "x" << sizeB.height << endl;
            // cout << "Required rotation for piece " << a << ": " << rotationA << " degrees" << endl;
            
            if (rotationA == 90.0f || rotationA == 270.0f) {
                sizeA = cv::Size(sizeA.height, sizeA.width);
                // cout << "Adjusted piece " << a << " size after rotation: " << sizeA.width << "x" << sizeA.height << endl;
            }
            
            cv::Point2f posB = positions[b].position;
            cv::Point2f offset(0.0f, 0.0f);
            switch (edgeB) {
                case 0: 
                    offset = cv::Point2f(0.0f, -sizeA.height); 
                    //  cout << "Edge configuration: B.top <-> A.bottom, placing A above B" << endl;
                    break;
                case 1: 
                    offset = cv::Point2f(sizeB.width, 0.0f);
                    // cout << "Edge configuration: B.right <-> A.left, placing A right of B" << endl;
                    break;
                case 2: 
                    offset = cv::Point2f(0.0f, sizeB.height);
                    // cout << "Edge configuration: B.bottom <-> A.top, placing A below B" << endl;
                    break;
                case 3: 
                    offset = cv::Point2f(-sizeA.width, 0.0f);
                    // cout << "Edge configuration: B.left <-> A.right, placing A left of B" << endl;
                    break;
            }
            
            cv::Point2f posA = posB + offset;
            positions[a] = {posA, rotationA, sizeA};

            int groupB = groupId[b];
            groupId[a] = groupB;
            groups[groupB].push_back(a);

            edgeUsed[a][edgeA] = true;
            edgeUsed[b][edgeB] = true;

            // cout << "Attached piece " << a << " at [" << posA.x << ", " << posA.y << "] to group " << groupB << endl;
            continue;
        }

        if (placedA && placedB) {
            // cout << "CASE: Both pieces placed, checking group connection" << endl;
            
            int groupA = groupId[a];
            int groupB = groupId[b];
            
            // cout << "Piece " << a << " in group " << groupA << endl;
            // cout << "Piece " << b << " in group " << groupB << endl;
            
            if (groupA != groupB) {
                cv::Point2f posA = positions[a].position;
                cv::Point2f posB = positions[b].position;
                cv::Size sizeA = positions[a].size;
                cv::Size sizeB = positions[b].size;
                
                // cout << "Piece " << a << " position: [" << posA.x << ", " << posA.y << "], size: " << sizeA.width << "x" << sizeA.height << endl;
                // cout << "Piece " << b << " position: [" << posB.x << ", " << posB.y << "], size: " << sizeB.width << "x" << sizeB.height << endl;

                cv::Point2f desiredOffset(0.0f, 0.0f);

                if ((edgeA == 0 && edgeB == 2) || (edgeA == 2 && edgeB == 0)) {
                    if (edgeA == 0) {
                        desiredOffset = cv::Point2f(0.0f, -sizeB.height);
                    } else {
                        desiredOffset = cv::Point2f(0.0f, sizeA.height);
                    }
                    // cout << "Vertical connection: B should be " << (edgeA == 0 ? "above" : "below") << " A" << endl;
                } else if ((edgeA == 1 && edgeB == 3) || (edgeA == 3 && edgeB == 1)) {
                    if (edgeA == 1) {
                        desiredOffset = cv::Point2f(sizeA.width, 0.0f);
                    } else {
                        desiredOffset = cv::Point2f(-sizeB.width, 0.0f);
                    }
                    // cout << "Horizontal connection: B should be to the " << (edgeA == 1 ? "right" : "left") << " of A" << endl;
                } else {
                    // This shouldn't happen for valid matches, but handle it
                    // cout << "WARNING: Non-complementary edges in match!" << endl;
                    switch (edgeA) {
                        case 0: 
                            desiredOffset = cv::Point2f(0.0f, -sizeB.height); 
                            break;
                        case 1: 
                            desiredOffset = cv::Point2f(sizeA.width, 0.0f);
                            break;
                        case 2: 
                            desiredOffset = cv::Point2f(0.0f, sizeA.height);
                            break;
                        case 3: 
                            desiredOffset = cv::Point2f(-sizeB.width, 0.0f);
                            break;
                    }
                }
                
                cv::Point2f currentOffset = posB - posA;
                cv::Point2f gapOffset = desiredOffset - currentOffset;
                
                // cout << "Desired offset B->A: [" << desiredOffset.x << ", " << desiredOffset.y << "]" << endl;
                // cout << "Current offset B->A: [" << currentOffset.x << ", " << currentOffset.y << "]" << endl;
                // cout << "Gap offset to fix: [" << gapOffset.x << ", " << gapOffset.y << "]" << endl;
                
                moveGroup(groupB, gapOffset);
                int mergedGroup = mergeGroups(groupA, groupB);
                
                edgeUsed[a][edgeA] = true;
                edgeUsed[b][edgeB] = true;
                
                //  cout << "Connected groups " << groupA << " and " << groupB << " into group " << mergedGroup << endl;
            } else {
                // cout << "SKIP: Both pieces already in same group " << groupA << endl;
            }
            continue;
        }
    }

    cout << "\nFINAL PLACEMENT STATUS" << endl;
    vector<int> missingPieces;
    for (int i = 0; i < N; i++) {
        if (positions.count(i)) {
            // cout << "Piece " << i << ": placed at [" << positions[i].position.x << ", " 
            //      << positions[i].position.y << "]" << endl;
        } else {
            // cout << "Piece " << i << ": MISSING" << endl;
            missingPieces.push_back(i);
        }
    }

    if (!missingPieces.empty()) {
        // cout << "\nPlacing " << missingPieces.size() << " missing pieces in grid" << endl;
        
        // calculate boundaries of placed pieces
        float startX = 0;
        float startY = 0;
        if (!positions.empty()) {
            cv::Rect2f existingBounds = findTotalArea(positions);
            startX = 0;
            startY = existingBounds.y + existingBounds.height + margin * 2;
        }
        
        float currentX = startX;
        float currentY = startY;
        float maxHeightInRow = 0;
        
        for (size_t i = 0; i < missingPieces.size(); i++) {
            int pieceId = missingPieces[i];
            cv::Size size = f[pieceId].img.size();
            
            positions[pieceId] = {cv::Point2f(currentX, currentY), 0.0f, size};
            
            // cout << "Placed missing piece " << pieceId << " at [" << currentX << ", " << currentY << "]" << endl;
            
            currentX += size.width + margin;
            maxHeightInRow = max(maxHeightInRow, (float)size.height);
            
            // if exceeds canvas width, new line
            if (i < missingPieces.size() - 1) {
                cv::Size nextSize = f[missingPieces[i+1]].img.size();
                if (currentX + nextSize.width > canvasW) {
                    currentX = startX;
                    currentY += maxHeightInRow + margin;
                    maxHeightInRow = 0;
                }
            }
        }
    }

    layout.positions = positions;
    layout.bounds = findTotalArea(positions);
    
    // cout << "\nLAYOUT COMPLETE" << endl;
    // cout << "Total pieces placed: " << layout.positions.size() << "/" << N << endl;
    // cout << "Layout bounds: [" << layout.bounds.x << ", " << layout.bounds.y << ", " 
    //      << layout.bounds.width << ", " << layout.bounds.height << "]" << endl;
    
    // verify: check for overlap or anomalies
    for (const auto& entry1 : positions) {
        for (const auto& entry2 : positions) {
            if (entry1.first >= entry2.first) continue;
            
            cv::Rect2f rect1(entry1.second.position.x, entry1.second.position.y,
                           entry1.second.size.width, entry1.second.size.height);
            cv::Rect2f rect2(entry2.second.position.x, entry2.second.position.y,
                           entry2.second.size.width, entry2.second.size.height);
            
            cv::Rect2f intersection = rect1 & rect2;
            if (intersection.area() > 100) {  // if overlap area is greater than 100 pixels
                cout << "WARNING: Pieces " << entry1.first << " and " << entry2.first 
                     << " have significant overlap (" << intersection.area() << " pixels)" << endl;
            }
        }
    }
    
    return layout;
}

// Raster scan
PuzzleLayout buildLayoutRasterScan(const vector<PieceFeature>& features, int canvasW, int canvasH) {
    int N = features.size();
    if (N == 0) {
        PuzzleLayout empty;
        return empty;
    }
    
    cout << "\n=== RASTER SCAN LAYOUT BUILDING ===" << endl;
    cout << "Total pieces: " << N << endl;
    
    // check if piece sizes are similar
    int avgWidth = 0, avgHeight = 0;
    int minWidth = INT_MAX, maxWidth = 0;
    int minHeight = INT_MAX, maxHeight = 0;
    for (const auto& f : features) {
        int w = f.img.cols;
        int h = f.img.rows;
        avgWidth += w;
        avgHeight += h;
        minWidth = min(minWidth, w);
        maxWidth = max(maxWidth, w);
        minHeight = min(minHeight, h);
        maxHeight = max(maxHeight, h);
    }
    avgWidth /= N;
    avgHeight /= N;
    
    bool similarSizes = (maxWidth - minWidth) < avgWidth * 0.2 && 
                        (maxHeight - minHeight) < avgHeight * 0.2;
    
    cout << "Piece sizes: avg=" << avgWidth << "x" << avgHeight 
         << ", range=[" << minWidth << "-" << maxWidth << "]x[" 
         << minHeight << "-" << maxHeight << "]" << endl;
    cout << "Similar sizes: " << (similarSizes ? "YES" : "NO") << endl;
    
    // get all possible grid shapes (rows × cols = N), but exclude long strip shapes
    vector<pair<int, int>> gridShapes;
    const double MAX_ASPECT_RATIO = 2.0;  // max aspect ratio, if greater than this, it is a long strip
    
    for (int r = 1; r <= N; r++) {
        if (N % r == 0) {
            int c = N / r;
            double aspectRatio = max((double)r / c, (double)c / r);
            
            // only keep shapes that are close to square (exclude long strips)
            if (aspectRatio <= MAX_ASPECT_RATIO) {
                gridShapes.push_back({r, c});
            } else {
                cout << "  Skipping long strip shape: " << r << "x" << c 
                     << " (aspect ratio: " << aspectRatio << ")" << endl;
            }
        }
    }
    
    if (gridShapes.empty()) {
        cerr << "ERROR: No valid grid shapes found! All shapes are too long." << endl;
        cerr << "Falling back to original method..." << endl;
        vector<Pair> matches = createFilteredMatches(features, 0.9);
        return buildLayout(matches, features, canvasW, canvasH);
    }
    
    // sort by aspect ratio, try to find the closest square shape first
    sort(gridShapes.begin(), gridShapes.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        double ratioA = max((double)a.first / a.second, (double)a.second / a.first);
        double ratioB = max((double)b.first / b.second, (double)b.second / b.first);
        return ratioA < ratioB;
    });
    
    cout << "Valid grid shapes (excluding long strips): ";
    for (const auto& shape : gridShapes) {
        double ratio = max((double)shape.first / shape.second, (double)shape.second / shape.first);
        cout << shape.first << "x" << shape.second << "(ratio=" << fixed << setprecision(2) << ratio << ") ";
    }
    cout << endl;
    
    if (gridShapes.empty()) {
        cerr << "ERROR: No valid grid shapes found after filtering!" << endl;
        cerr << "All possible shapes are too long. Falling back to original method..." << endl;
        vector<Pair> matches = createFilteredMatches(features, 0.9);
        return buildLayout(matches, features, canvasW, canvasH);
    }
    
    PuzzleLayout bestLayout;
    double bestScore = numeric_limits<double>::max();
    
    for (const auto& shape : gridShapes) {
        int rows = shape.first;
        int cols = shape.second;
        
        cout << "\nTrying grid shape: " << rows << "x" << cols << endl;
        
        // try each piece as a seed (starting position (0,0))
        for (int seedId = 0; seedId < N; seedId++) {
            // try each rotation angle (0, 90, 180, 270)
            for (int seedRotation = 0; seedRotation < 4; seedRotation++) {
                // create grid
                vector<vector<int>> grid(rows, vector<int>(cols, -1));
                unordered_map<int, PiecePosition> positions;
                unordered_set<int> usedPieces;
                
                if (cols <= 0 || rows <= 0) {
                    cerr << "ERROR: Invalid grid size: " << rows << "x" << cols << endl;
                    continue;
                }
                vector<float> colWidths(cols, 0);
                vector<float> rowHeights(rows, 0);
                vector<float> xOffsets(cols, 0);
                vector<float> yOffsets(rows, 0);
                
                if (colWidths.size() != cols || rowHeights.size() != rows || 
                    xOffsets.size() != cols || yOffsets.size() != rows) {
                    cerr << "ERROR: Vector size mismatch!" << endl;
                    continue;
                }
                
                grid[0][0] = seedId;
                usedPieces.insert(seedId);
                cv::Size seedSize = features[seedId].img.size();
                if (seedRotation == 1 || seedRotation == 3) {
                    seedSize = cv::Size(seedSize.height, seedSize.width);
                }
                positions[seedId] = {cv::Point2f(0, 0), seedRotation * 90.0f, seedSize};
                
                colWidths[0] = seedSize.width;
                rowHeights[0] = seedSize.height;
                
                bool valid = true;
                double totalScore = 0.0;
                int numEdges = 0;
                
                // raster scan fill
                for (int r = 0; r < rows && valid; r++) {
                    for (int c = 0; c < cols && valid; c++) {
                        if (r == 0 && c == 0) continue;
                        
                        int leftNeighbor = (c > 0) ? grid[r][c-1] : -1;
                        int topNeighbor = (r > 0) ? grid[r-1][c] : -1;
                        
                        if (leftNeighbor == -1 && topNeighbor == -1) {
                            valid = false;
                            break;
                        }
                        
                        int bestPiece = -1;
                        int bestRotation = 0;
                        double bestScore = numeric_limits<double>::max();
                        
                        for (int pieceId = 0; pieceId < N; pieceId++) {
                            if (usedPieces.count(pieceId)) continue;
                            
                            for (int rot = 0; rot < 4; rot++) {
                                double score = 0.0;
                                bool hasMatch = false;
                                
                                if (leftNeighbor >= 0) {
                                    double matchScore = edgeDistanceFull(
                                        features[leftNeighbor], features[pieceId],
                                        1, 3 
                                    );
                                    score += matchScore;
                                    hasMatch = true;
                                    numEdges++;
                                }

                                if (topNeighbor >= 0) {
                                    double matchScore = edgeDistanceFull(
                                        features[topNeighbor], features[pieceId],
                                        2, 0
                                    );
                                    score += matchScore;
                                    hasMatch = true;
                                    numEdges++;
                                }
                                
                                if (hasMatch && score < bestScore) {
                                    bestScore = score;
                                    bestPiece = pieceId;
                                    bestRotation = rot;
                                }
                            }
                        }
                        
                        if (bestPiece == -1) {
                            valid = false;
                            break;
                        }
                        
                        grid[r][c] = bestPiece;
                        usedPieces.insert(bestPiece);
                        
                        cv::Size pieceSize = features[bestPiece].img.size();
                        if (bestRotation == 1 || bestRotation == 3) {
                            pieceSize = cv::Size(pieceSize.height, pieceSize.width);
                        }
                        
                        if (c < 0 || c >= cols || r < 0 || r >= rows) {
                            cerr << "ERROR: Grid index out of bounds! r=" << r << "/" << rows 
                                 << ", c=" << c << "/" << cols << endl;
                            valid = false;
                            break;
                        }
                        
                        if (colWidths.size() != cols || rowHeights.size() != rows || 
                            xOffsets.size() != cols || yOffsets.size() != rows) {
                            cerr << "ERROR: Vector size mismatch! cols=" << cols << ", rows=" << rows 
                                 << ", colWidths=" << colWidths.size() << ", rowHeights=" << rowHeights.size() << endl;
                            valid = false;
                            break;
                        }
                        
                        colWidths[c] = max(colWidths[c], (float)pieceSize.width);
                        rowHeights[r] = max(rowHeights[r], (float)pieceSize.height);
                        
                        if (cols > 0 && xOffsets.size() == cols && colWidths.size() == cols) {
                            for (int ci = 1; ci < cols; ci++) {
                                if (ci < (int)xOffsets.size() && ci-1 < (int)colWidths.size() && ci-1 < (int)xOffsets.size()) {
                                    xOffsets[ci] = xOffsets[ci-1] + colWidths[ci-1];
                                }
                            }
                        }
                        if (rows > 0 && yOffsets.size() == rows && rowHeights.size() == rows) {
                            for (int ri = 1; ri < rows; ri++) {
                                if (ri < (int)yOffsets.size() && ri-1 < (int)rowHeights.size() && ri-1 < (int)yOffsets.size()) {
                                    yOffsets[ri] = yOffsets[ri-1] + rowHeights[ri-1];
                                }
                            }
                        }
                        
                        for (int ri = 0; ri < rows; ri++) {
                            for (int ci = 0; ci < cols; ci++) {
                                if (grid[ri][ci] >= 0 && (ri < r || (ri == r && ci < c))) {
                                    int pid = grid[ri][ci];
                                    if (ci < (int)xOffsets.size() && ri < (int)yOffsets.size()) {
                                        positions[pid].position.x = xOffsets[ci];
                                        positions[pid].position.y = yOffsets[ri];
                                    }
                                }
                            }
                        }

                        if (c >= (int)xOffsets.size() || r >= (int)yOffsets.size()) {
                            cerr << "ERROR: Final offset access out of bounds! c=" << c 
                                 << "/" << xOffsets.size() << ", r=" << r << "/" << yOffsets.size() << endl;
                            valid = false;
                            break;
                        }
                        cv::Point2f pos(xOffsets[c], yOffsets[r]);
                        
                        positions[bestPiece] = {pos, bestRotation * 90.0f, pieceSize};
                        totalScore += bestScore;
                    }
                }
                
                // if all positions are filled, calculate average score and check global consistency
                if (valid && usedPieces.size() == N) {
                    double avgScore = (numEdges > 0) ? totalScore / numEdges : totalScore;
                    
                    if (avgScore < bestScore) {
                        bestScore = avgScore;
                        bestLayout.grid = grid;
                        bestLayout.positions = positions;
                        bestLayout.rows = rows;
                        bestLayout.cols = cols;
                        bestLayout.bounds = findTotalArea(positions);
                        
                        cout << "  Found better solution: seed=" << seedId 
                             << ", rotation=" << seedRotation * 90
                             << ", avgScore=" << avgScore << endl;
                    }
                }
            }
        }
    }
    
    if (bestLayout.positions.empty()) {
        cout << "WARNING: Raster scan failed, falling back to original method" << endl;
        // fall back to original method
        vector<Pair> matches = createFilteredMatches(features, 0.9);
        return buildLayout(matches, features, canvasW, canvasH);
    }
    
    cout << "\nBest solution found with avgScore=" << bestScore << endl;
    cout << "Grid size: " << bestLayout.rows << "x" << bestLayout.cols << endl;
    
    return bestLayout;
}

}