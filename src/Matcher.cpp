#include "Matcher.h"
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

double computeEdgeProfileCompatibility(const EdgeFeature& efA, const EdgeFeature& efB) {
    
    vector<double> reversedB = efB.vals;
    reverse(reversedB.begin(), reversedB.end());
    
    double forwardMatch = 0.0, reverseMatch = 0.0;
    double forwardSlopeConsistency = 0.0, reverseSlopeConsistency = 0.0;
    
    for (int i = 0; i < efA.vals.size(); i++) {
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

    double L = distLuma(efA, efB);
    double P = computeEdgeProfileCompatibility(efA, efB);
    double T = computeTextureConsistency(A.img, B.img, edgeA, edgeB);

    double dist = 
        1.5 * L +        // Luma structure
        0.5 * C +        // Color consistency  
        1.0 * G +        // Gradient strength
        4.0 * P +        // Profile compatibility
        2.0 * T;        // Texture consistency

 

    return dist;
}

namespace Matcher {
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
    
    int checkFirstN = min(20, (int)matches.size());
    double avgTopScores = 0;
    for (int i = 0; i < checkFirstN; i++) {
        avgTopScores += matches[i].val;
    }
    avgTopScores /= checkFirstN;
    
    double scoreCutoff = avgTopScores * 4.0;
    
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
    cout << "Total pieces: " << N << ", Total matches: " << matches.size() << endl;

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
        
        cout << "\nProcessing match " << i << " ===" << endl;
        cout << "Piece " << a << " (edge " << edgeA << ") <-> Piece " << b << " (edge " << edgeB << "), Score: " << m.val << endl;
        
        if (allPiecesConnected()) {
            cout << "All pieces placed, stopping." << endl;
            break;
        }
        
        if (edgeUsed[a][edgeA]) {
            cout << "SKIP: Edge " << edgeA << " of piece " << a << " already used." << endl;
            continue;
        }
        if (edgeUsed[b][edgeB]) {
            cout << "SKIP: Edge " << edgeB << " of piece " << b << " already used." << endl;
            continue;
        }

        bool placedA = positions.count(a) != 0;
        bool placedB = positions.count(b) != 0;

        cout << "Piece " << a << " placed: " << (placedA ? "YES" : "NO") << endl;
        cout << "Piece " << b << " placed: " << (placedB ? "YES" : "NO") << endl;

        if (!placedA && !placedB) {
            cout << "CASE: Neither piece placed - creating new group" << endl;
            
            float rotationB = calculateRequiredRotation(edgeA, edgeB);
            cv::Size sizeA = f[a].img.size();
            cv::Size sizeB = f[b].img.size();
            
            cout << "Piece " << a << " size: " << sizeA.width << "x" << sizeA.height << endl;
            cout << "Piece " << b << " size: " << sizeB.width << "x" << sizeB.height << endl;
            cout << "Required rotation for piece " << b << ": " << rotationB << " degrees" << endl;
            

            if (rotationB == 90.0f || rotationB == 270.0f) {
                sizeB = cv::Size(sizeB.height, sizeB.width);
                cout << "Adjusted piece " << b << " size after rotation: " << sizeB.width << "x" << sizeB.height << endl;
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
                    cout << "Edge configuration: A.top <-> B.bottom, placing B above A" << endl;
                    break;
                case 1: 
                    offset = cv::Point2f((float)sizeA.width, 0.0f);
                    cout << "Edge configuration: A.right <-> B.left, placing B right of A" << endl;
                    break;
                case 2: 
                    offset = cv::Point2f(0.0f, (float)sizeA.height);
                    cout << "Edge configuration: A.bottom <-> B.top, placing B below A" << endl;
                    break;
                case 3: 
                    offset = cv::Point2f(- (float)sizeB.width, 0.0f);
                    cout << "Edge configuration: A.left <-> B.right, placing B left of A" << endl;
                    break;
            }
            
            cv::Point2f posB = posA + offset;

            positions[a] = {posA, 0.0f, sizeA};
            positions[b] = {posB, rotationB, sizeB};

            edgeUsed[a][edgeA] = true;
            edgeUsed[b][edgeB] = true;

            cout << "Placed piece " << a << " at [" << posA.x << ", " << posA.y << "]" << endl;
            cout << "Placed piece " << b << " at [" << posB.x << ", " << posB.y << "]" << endl;

            cv::Rect2f groupBounds = getGroupBounds(newGroup);
            cout << "Group " << newGroup << " bounds: [" << groupBounds.x << ", " << groupBounds.y << ", " 
                 << groupBounds.width << ", " << groupBounds.height << "]" << endl;
                 
            newGroupXCoordinate += (int)groupBounds.width + margin;
            cout << "Next group X coordinate: " << newGroupXCoordinate << endl;
            continue;
        }

        if (placedA && !placedB) {
            cout << "CASE: Piece " << a << " placed, attaching piece " << b << endl;
            
            float rotationB = calculateRequiredRotation(edgeA, edgeB);
            cv::Size sizeA = positions[a].size;
            cv::Size sizeB = f[b].img.size();
            
            cout << "Piece " << a << " current size: " << sizeA.width << "x" << sizeA.height << endl;
            cout << "Piece " << b << " original size: " << sizeB.width << "x" << sizeB.height << endl;
            cout << "Required rotation for piece " << b << ": " << rotationB << " degrees" << endl;
            
            if (rotationB == 90.0f || rotationB == 270.0f) {
                sizeB = cv::Size(sizeB.height, sizeB.width);
                cout << "Adjusted piece " << b << " size after rotation: " << sizeB.width << "x" << sizeB.height << endl;
            }
            
            cv::Point2f posA = positions[a].position;
            cv::Point2f offset(0.0f, 0.0f);
            switch (edgeA) {
                case 0: 
                    offset = cv::Point2f(0.0f, -sizeB.height); 
                    cout << "Edge configuration: A.top <-> B.bottom, placing B above A" << endl;
                    break;
                case 1: 
                    offset = cv::Point2f(sizeA.width, 0.0f);
                    cout << "Edge configuration: A.right <-> B.left, placing B right of A" << endl;
                    break;
                case 2: 
                    offset = cv::Point2f(0.0f, sizeA.height);
                    cout << "Edge configuration: A.bottom <-> B.top, placing B below A" << endl;
                    break;
                case 3: 
                    offset = cv::Point2f(-sizeB.width, 0.0f);
                    cout << "Edge configuration: A.left <-> B.right, placing B left of A" << endl;
                    break;
            }
            
            cv::Point2f posB = posA + offset;
            positions[b] = {posB, rotationB, sizeB};

            int groupA = groupId[a];
            groupId[b] = groupA;
            groups[groupA].push_back(b);

            edgeUsed[a][edgeA] = true;
            edgeUsed[b][edgeB] = true;

            cout << "Attached piece " << b << " at [" << posB.x << ", " << posB.y << "] to group " << groupA << endl;
            continue;
        }

        if (!placedA && placedB) {
            cout << "CASE: Piece " << b << " placed, attaching piece " << a << endl;
            
            float rotationA = calculateRequiredRotation(edgeA, edgeB);
            cv::Size sizeA = f[a].img.size();
            cv::Size sizeB = positions[b].size;
            
            cout << "Piece " << a << " original size: " << sizeA.width << "x" << sizeA.height << endl;
            cout << "Piece " << b << " current size: " << sizeB.width << "x" << sizeB.height << endl;
            cout << "Required rotation for piece " << a << ": " << rotationA << " degrees" << endl;
            
            if (rotationA == 90.0f || rotationA == 270.0f) {
                sizeA = cv::Size(sizeA.height, sizeA.width);
                cout << "Adjusted piece " << a << " size after rotation: " << sizeA.width << "x" << sizeA.height << endl;
            }
            
            cv::Point2f posB = positions[b].position;
            cv::Point2f offset(0.0f, 0.0f);
            switch (edgeB) {
                case 0: 
                    offset = cv::Point2f(0.0f, -sizeA.height); 
                    cout << "Edge configuration: B.top <-> A.bottom, placing A above B" << endl;
                    break;
                case 1: 
                    offset = cv::Point2f(sizeB.width, 0.0f);
                    cout << "Edge configuration: B.right <-> A.left, placing A right of B" << endl;
                    break;
                case 2: 
                    offset = cv::Point2f(0.0f, sizeB.height);
                    cout << "Edge configuration: B.bottom <-> A.top, placing A below B" << endl;
                    break;
                case 3: 
                    offset = cv::Point2f(-sizeA.width, 0.0f);
                    cout << "Edge configuration: B.left <-> A.right, placing A left of B" << endl;
                    break;
            }
            
            cv::Point2f posA = posB + offset;
            positions[a] = {posA, rotationA, sizeA};

            int groupB = groupId[b];
            groupId[a] = groupB;
            groups[groupB].push_back(a);

            edgeUsed[a][edgeA] = true;
            edgeUsed[b][edgeB] = true;

            cout << "Attached piece " << a << " at [" << posA.x << ", " << posA.y << "] to group " << groupB << endl;
            continue;
        }

        if (placedA && placedB) {
            cout << "CASE: Both pieces placed, checking group connection" << endl;
            
            int groupA = groupId[a];
            int groupB = groupId[b];
            
            cout << "Piece " << a << " in group " << groupA << endl;
            cout << "Piece " << b << " in group " << groupB << endl;
            
            if (groupA != groupB) {
                cv::Point2f posA = positions[a].position;
                cv::Point2f posB = positions[b].position;
                cv::Size sizeA = positions[a].size;
                cv::Size sizeB = positions[b].size;
                
                cout << "Piece " << a << " position: [" << posA.x << ", " << posA.y << "], size: " << sizeA.width << "x" << sizeA.height << endl;
                cout << "Piece " << b << " position: [" << posB.x << ", " << posB.y << "], size: " << sizeB.width << "x" << sizeB.height << endl;

                cv::Point2f desiredOffset(0.0f, 0.0f);

                if ((edgeA == 0 && edgeB == 2) || (edgeA == 2 && edgeB == 0)) {
                    if (edgeA == 0) {
                        desiredOffset = cv::Point2f(0.0f, -sizeB.height);
                    } else {
                        desiredOffset = cv::Point2f(0.0f, sizeA.height);
                    }
                    cout << "Vertical connection: B should be " << (edgeA == 0 ? "above" : "below") << " A" << endl;
                } else if ((edgeA == 1 && edgeB == 3) || (edgeA == 3 && edgeB == 1)) {
                    if (edgeA == 1) {
                        desiredOffset = cv::Point2f(sizeA.width, 0.0f);
                    } else {
                        desiredOffset = cv::Point2f(-sizeB.width, 0.0f);
                    }
                    cout << "Horizontal connection: B should be to the " << (edgeA == 1 ? "right" : "left") << " of A" << endl;
                } else {
                    // This shouldn't happen for valid matches, but handle it
                    cout << "WARNING: Non-complementary edges in match!" << endl;
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
                
                cout << "Desired offset B->A: [" << desiredOffset.x << ", " << desiredOffset.y << "]" << endl;
                cout << "Current offset B->A: [" << currentOffset.x << ", " << currentOffset.y << "]" << endl;
                cout << "Gap offset to fix: [" << gapOffset.x << ", " << gapOffset.y << "]" << endl;
                
                moveGroup(groupB, gapOffset);
                int mergedGroup = mergeGroups(groupA, groupB);
                
                edgeUsed[a][edgeA] = true;
                edgeUsed[b][edgeB] = true;
                
                cout << "Connected groups " << groupA << " and " << groupB << " into group " << mergedGroup << endl;
            } else {
                cout << "SKIP: Both pieces already in same group " << groupA << endl;
            }
            continue;
        }
    }

    cout << "\nFINAL PLACEMENT STATUS" << endl;
    for (int i = 0; i < N; i++) {
        if (positions.count(i)) {
            cout << "Piece " << i << ": placed at [" << positions[i].position.x << ", " << positions[i].position.y << "]" << endl;
        } else {
            cout << "Piece " << i << ": MISSING" << endl;
        }
    }

    // Place missing pieces at the bottom
    // vector<int> missingPieces;
    // for (int i = 0; i < N; i++) {
    //     if (positions.count(i) == 0) {
    //         missingPieces.push_back(i);
    //     }
    // }
    // if (!missingPieces.empty()) {
    //     cout << "\nPlacing " << missingPieces.size() << " missing pieces in grid" << endl;
    //     float startX = 0;
    //     float startY = layout.bounds.y + layout.bounds.height + margin;
    //     float currentX = startX;
    //     float currentY = startY;
    //     float maxHeightInRow = 0;
        
    //     for (int i = 0; i < missingPieces.size(); i++) {
    //         int pieceId = missingPieces[i];
    //         cv::Size size = f[pieceId].img.size();
            
    //         positions[pieceId] = {cv::Point2f(currentX, currentY), 0.0f, size};
            
    //         cout << "Placed missing piece " << pieceId << " at [" << currentX << ", " << currentY << "]" << endl;
            
    //         currentX += size.width + margin;
    //         maxHeightInRow = max(maxHeightInRow, (float)size.height);
            
    //         // Move to next row if we exceed width
    //         if (i < missingPieces.size() - 1 && currentX + f[missingPieces[i+1]].img.cols > canvasW) {
    //             currentX = startX;
    //             currentY += maxHeightInRow + margin;
    //             maxHeightInRow = 0;
    //         }
    //     }
    // }

    layout.positions = positions;
    layout.bounds = findTotalArea(positions);
    
    cout << "\nLAYOUT COMPLETE" << endl;
    cout << "Total pieces placed: " << layout.positions.size() << "/" << N << endl;
    cout << "Layout bounds: [" << layout.bounds.x << ", " << layout.bounds.y << ", " 
         << layout.bounds.width << ", " << layout.bounds.height << "]" << endl;
    
    return layout;
}

}