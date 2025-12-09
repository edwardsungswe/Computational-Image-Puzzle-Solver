#include "PieceMatcher.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <cstring>

using namespace std;
using namespace cv;

namespace {

// MGC (Mahalanobis Gradient Compatibility)
// 3x3 matrix operations for Mahalanobis distance
struct Mat3x3 {
    double data[3][3];
    
    Mat3x3() { memset(data, 0, sizeof(data)); }
    
    double& operator()(int i, int j) { return data[i][j]; }
    double operator()(int i, int j) const { return data[i][j]; }
    
    static Mat3x3 identity() {
        Mat3x3 m;
        m(0,0) = m(1,1) = m(2,2) = 1.0;
        return m;
    }
    
    Mat3x3 inverse() const {
        Mat3x3 inv;
        double det = data[0][0] * (data[1][1] * data[2][2] - data[1][2] * data[2][1])
                   - data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0])
                   + data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
        
        if (abs(det) < 1e-10) {
            return identity();
        }
        
        double invDet = 1.0 / det;
        
        inv(0,0) =  (data[1][1] * data[2][2] - data[1][2] * data[2][1]) * invDet;
        inv(0,1) = -(data[0][1] * data[2][2] - data[0][2] * data[2][1]) * invDet;
        inv(0,2) =  (data[0][1] * data[1][2] - data[0][2] * data[1][1]) * invDet;
        inv(1,0) = -(data[1][0] * data[2][2] - data[1][2] * data[2][0]) * invDet;
        inv(1,1) =  (data[0][0] * data[2][2] - data[0][2] * data[2][0]) * invDet;
        inv(1,2) = -(data[0][0] * data[1][2] - data[0][2] * data[1][0]) * invDet;
        inv(2,0) =  (data[1][0] * data[2][1] - data[1][1] * data[2][0]) * invDet;
        inv(2,1) = -(data[0][0] * data[2][1] - data[0][1] * data[2][0]) * invDet;
        inv(2,2) =  (data[0][0] * data[1][1] - data[0][1] * data[1][0]) * invDet;
        
        return inv;
    }
};

struct Vec3 {
    double x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    
    double mahalanobis(const Vec3& mu, const Mat3x3& sigmaInv) const {
        Vec3 d = *this - mu;
        double vals[3] = {d.x, d.y, d.z};
        double result = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result += vals[i] * sigmaInv(i, j) * vals[j];
            }
        }
        return sqrt(max(0.0, result));
    }
};

// Dummy gradients for numerical stability (from Gallagher's paper)
const Vec3 DUMMY_GRADIENTS[] = {
    Vec3(0, 0, 0),
    Vec3(1, 1, 1),
    Vec3(-1, -1, -1),
    Vec3(0, 0, 1),
    Vec3(0, 1, 0),
    Vec3(1, 0, 0),
    Vec3(-1, 0, 0),
    Vec3(0, -1, 0),
    Vec3(0, 0, -1)
};
const int NUM_DUMMIES = 9;

void computeMeanAndCovariance(const vector<Vec3>& samples, Vec3& mu, Mat3x3& cov) {
    int n = samples.size();
    if (n == 0) {
        mu = Vec3();
        cov = Mat3x3::identity();
        return;
    }
    
    // Mean
    mu = Vec3();
    for (const auto& s : samples) {
        mu += s;
    }
    mu = mu / n;
    
    // Covariance
    cov = Mat3x3();
    for (const auto& s : samples) {
        Vec3 d = s - mu;
        double vals[3] = {d.x, d.y, d.z};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cov(i, j) += vals[i] * vals[j];
            }
        }
    }
    
    // Normalize and add regularization
    double regularization = 1.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cov(i, j) /= max(1, n - 1);
        }
        cov(i, i) += regularization;
    }
}

double sumMahalanobis(const vector<Vec3>& samples, const Vec3& mu, const Mat3x3& sigmaInv) {
    double total = 0.0;
    for (const auto& s : samples) {
        total += s.mahalanobis(mu, sigmaInv);
    }
    return total;
}

// MGC for LEFT-RIGHT adjacency (pieceA on LEFT, pieceB on RIGHT)
double mgcCompatibilityLR(const Mat& pieceA, const Mat& pieceB) {
    int heightA = pieceA.rows;
    int heightB = pieceB.rows;
    int height = min(heightA, heightB);
    
    // Need at least 3 pixels of edge depth now
    if (height < 3 || pieceA.cols < 4 || pieceB.cols < 4) return 1e9;
    
    // Size compatibility check
    double ratio = (double)heightA / heightB;
    if (ratio < 0.8 || ratio > 1.25) return 1e9;
    
    vector<Vec3> G_L;
    vector<Vec3> G_R;
    vector<Vec3> G_LR;
    
    G_L.reserve(height + NUM_DUMMIES);
    G_R.reserve(height + NUM_DUMMIES);
    G_LR.reserve(height);
    
    // Skip edge pixels (offset inward by 1 to avoid interpolation artifacts)
    const int EDGE_SKIP = 1;
    
    for (int row = 0; row < height; row++) {
        int rowA = row * heightA / height;
        int rowB = row * heightB / height;
        
        // Sample 1 pixel inward from the actual edge
        Vec3b a_last = pieceA.at<Vec3b>(rowA, pieceA.cols - 1 - EDGE_SKIP);
        Vec3b a_prev = pieceA.at<Vec3b>(rowA, pieceA.cols - 2 - EDGE_SKIP);
        Vec3b b_first = pieceB.at<Vec3b>(rowB, 0 + EDGE_SKIP);
        Vec3b b_next = pieceB.at<Vec3b>(rowB, 1 + EDGE_SKIP);
        
        // G_L: gradient at right edge of A
        Vec3 gl(a_last[2] - a_prev[2],
                a_last[1] - a_prev[1],
                a_last[0] - a_prev[0]);
        G_L.push_back(gl);
        
        // G_R: gradient at left edge of B
        Vec3 gr(b_next[2] - b_first[2],
                b_next[1] - b_first[1],
                b_next[0] - b_first[0]);
        G_R.push_back(gr);
        
        // G_LR: gradient across the boundary
        Vec3 glr(b_first[2] - a_last[2],
                 b_first[1] - a_last[1],
                 b_first[0] - a_last[0]);
        G_LR.push_back(glr);
    }
    
    // Add dummy gradients
    for (int i = 0; i < NUM_DUMMIES; i++) {
        G_L.push_back(DUMMY_GRADIENTS[i]);
        G_R.push_back(DUMMY_GRADIENTS[i]);
    }
    
    Vec3 mu_L, mu_R;
    Mat3x3 cov_L, cov_R;
    computeMeanAndCovariance(G_L, mu_L, cov_L);
    computeMeanAndCovariance(G_R, mu_R, cov_R);
    
    Mat3x3 covInv_L = cov_L.inverse();
    Mat3x3 covInv_R = cov_R.inverse();
    
    double D_L = sumMahalanobis(G_LR, mu_L, covInv_L);
    double D_R = sumMahalanobis(G_LR, mu_R, covInv_R);
    
    return min(D_L, D_R);
}

// MGC for TOP-BOTTOM adjacency (pieceA on TOP, pieceB on BOTTOM)
double mgcCompatibilityTB(const Mat& pieceA, const Mat& pieceB) {
    int widthA = pieceA.cols;
    int widthB = pieceB.cols;
    int width = min(widthA, widthB);
    
    // Need at least 4 pixels of edge depth now
    if (width < 3 || pieceA.rows < 4 || pieceB.rows < 4) return 1e9;
    
    // Size compatibility check
    double ratio = (double)widthA / widthB;
    if (ratio < 0.8 || ratio > 1.25) return 1e9;
    
    vector<Vec3> G_T;
    vector<Vec3> G_B;
    vector<Vec3> G_TB;
    
    G_T.reserve(width + NUM_DUMMIES);
    G_B.reserve(width + NUM_DUMMIES);
    G_TB.reserve(width);
    
    // Skip edge pixels (offset inward by 1 to avoid interpolation artifacts)
    const int EDGE_SKIP = 1;
    
    for (int col = 0; col < width; col++) {
        int colA = col * widthA / width;
        int colB = col * widthB / width;
        
        // Sample 1 pixel inward from the actual edge
        Vec3b a_last = pieceA.at<Vec3b>(pieceA.rows - 1 - EDGE_SKIP, colA);
        Vec3b a_prev = pieceA.at<Vec3b>(pieceA.rows - 2 - EDGE_SKIP, colA);
        Vec3b b_first = pieceB.at<Vec3b>(0 + EDGE_SKIP, colB);
        Vec3b b_next = pieceB.at<Vec3b>(1 + EDGE_SKIP, colB);
        
        Vec3 gt(a_last[2] - a_prev[2],
                a_last[1] - a_prev[1],
                a_last[0] - a_prev[0]);
        G_T.push_back(gt);
        
        Vec3 gb(b_next[2] - b_first[2],
                b_next[1] - b_first[1],
                b_next[0] - b_first[0]);
        G_B.push_back(gb);
        
        Vec3 gtb(b_first[2] - a_last[2],
                 b_first[1] - a_last[1],
                 b_first[0] - a_last[0]);
        G_TB.push_back(gtb);
    }
    
    for (int i = 0; i < NUM_DUMMIES; i++) {
        G_T.push_back(DUMMY_GRADIENTS[i]);
        G_B.push_back(DUMMY_GRADIENTS[i]);
    }
    
    Vec3 mu_T, mu_B;
    Mat3x3 cov_T, cov_B;
    computeMeanAndCovariance(G_T, mu_T, cov_T);
    computeMeanAndCovariance(G_B, mu_B, cov_B);
    
    Mat3x3 covInv_T = cov_T.inverse();
    Mat3x3 covInv_B = cov_B.inverse();
    
    double D_T = sumMahalanobis(G_TB, mu_T, covInv_T);
    double D_B = sumMahalanobis(G_TB, mu_B, covInv_B);
    
    return min(D_T, D_B);
}

// Rotation helpers

Mat rotateImage90(const Mat& img, int steps) {
    steps = ((steps % 4) + 4) % 4;
    if (steps == 0) return img.clone();
    Mat result;
    if (steps == 1) { transpose(img, result); flip(result, result, 0); }
    else if (steps == 2) { flip(img, result, -1); }
    else { transpose(img, result); flip(result, result, 1); }
    return result;
}

Size getRotatedSize(const Mat& img, int rotSteps) {
    rotSteps = ((rotSteps % 4) + 4) % 4;
    return (rotSteps % 2 == 0) ? img.size() : Size(img.rows, img.cols);
}

// Grid shape inference
pair<int, int> inferGridShape(const vector<PieceFeature>& features, int N, int canvasW, int canvasH) {
    if (N == 0) return {1, 1};
    
    float avgW = 0, avgH = 0;
    for (const auto& f : features) {
        avgW += f.img.cols;
        avgH += f.img.rows;
    }
    avgW /= N;
    avgH /= N;
    
    float canvasAspect = (float)canvasW / canvasH;
    
    pair<int, int> best = {1, N};
    float bestScore = 1e9;
    
    for (int r = 1; r <= N; r++) {
        if (N % r != 0) continue;
        int c = N / r;
        float gridAspect = (c * avgW) / (r * avgH);
        float score = abs(log(gridAspect / canvasAspect));
        if (score < bestScore) {
            bestScore = score;
            best = {r, c};
        }
    }
    
    return best;
}

// Edge cache

using EdgeCache = vector<vector<vector<vector<double>>>>;

void buildMGCCache(const vector<PieceFeature>& features,
                   EdgeCache& hCache, EdgeCache& vCache) {
    int N = features.size();
    hCache.assign(N, vector<vector<vector<double>>>(N, vector<vector<double>>(4, vector<double>(4, 1e9))));
    vCache.assign(N, vector<vector<vector<double>>>(N, vector<vector<double>>(4, vector<double>(4, 1e9))));
    
    cout << "  Computing MGC scores for " << N << " pieces..." << endl;
    
    for (int a = 0; a < N; a++) {
        if ((a + 1) % 5 == 0 || a == N - 1) {
            cout << "    Piece " << (a + 1) << "/" << N << endl;
        }
        
        for (int b = 0; b < N; b++) {
            if (a == b) continue;
            
            for (int ra = 0; ra < 4; ra++) {
                Mat rotA = rotateImage90(features[a].img, ra);
                
                for (int rb = 0; rb < 4; rb++) {
                    Mat rotB = rotateImage90(features[b].img, rb);
                    
                    hCache[a][b][ra][rb] = mgcCompatibilityLR(rotA, rotB);
                    vCache[a][b][ra][rb] = mgcCompatibilityTB(rotA, rotB);
                }
            }
        }
    }
}

    
struct SearchState {
    vector<vector<int>> grid;
    vector<int> rotations;
    vector<bool> used;
    double score;
    int filledCount;
    
    bool operator<(const SearchState& other) const {
        return score > other.score;
    }
};

}  // anonymous namespace

namespace PieceMatcher {

Mat rotatePiece(const Mat& img, float rotationDegrees) {
    int steps = static_cast<int>(round(rotationDegrees / 90.0f)) % 4;
    if (steps < 0) steps += 4;
    return rotateImage90(img, steps);
}

PuzzleLayout solve(const vector<PieceFeature>& features, int canvasW, int canvasH) {
    int N = static_cast<int>(features.size());
    PuzzleLayout emptyLayout;
    if (N == 0) return emptyLayout;

    cout << "\n=== MGC PUZZLE SOLVER ===" << endl;
    cout << "Pieces: " << N << endl;

    auto gridShape = inferGridShape(features, N, canvasW, canvasH);
    int rows = gridShape.first;
    int cols = gridShape.second;
    cout << "Grid: " << rows << "x" << cols << endl;

    cout << "Building MGC edge caches..." << endl;
    EdgeCache hCache, vCache;
    buildMGCCache(features, hCache, vCache);

    const int BEAM_WIDTH = 5000;
    cout << "Beam search (width=" << BEAM_WIDTH << ")..." << endl;

    priority_queue<SearchState> beam;
    
    for (int seed = 0; seed < N; seed++) {
        SearchState state;
        state.grid.assign(rows, vector<int>(cols, -1));
        state.rotations.assign(N, 0);
        state.used.assign(N, false);
        state.grid[0][0] = seed;
        state.rotations[seed] = 0;
        state.used[seed] = true;
        state.score = 0.0;
        state.filledCount = 1;
        beam.push(state);
    }

    vector<SearchState> currentBeam;
    while (!beam.empty() && currentBeam.size() < BEAM_WIDTH) {
        currentBeam.push_back(beam.top());
        beam.pop();
    }

    SearchState bestComplete;
    bestComplete.filledCount = 0;
    bestComplete.score = 1e9;

    for (int step = 1; step < N; step++) {
        int nextRow = step / cols;
        int nextCol = step % cols;
        
        if (step % 5 == 0) {
            cout << "  Step " << step << "/" << N 
                 << ", beam size: " << currentBeam.size() 
                 << ", best score: " << currentBeam[0].score << endl;
        }

        priority_queue<SearchState> nextBeam;

        for (const auto& state : currentBeam) {
            for (int pid = 0; pid < N; pid++) {
                if (state.used[pid]) continue;

                for (int rot = 0; rot < 4; rot++) {
                    double score = 0.0;
                    bool valid = true;
                    
                    if (nextCol > 0) {
                        int leftPiece = state.grid[nextRow][nextCol - 1];
                        int leftRot = state.rotations[leftPiece];
                        double h = hCache[leftPiece][pid][leftRot][rot];
                        if (h >= 1e8) { valid = false; }
                        else score += h;
                    }
                    
                    if (valid && nextRow > 0) {
                        int topPiece = state.grid[nextRow - 1][nextCol];
                        int topRot = state.rotations[topPiece];
                        double v = vCache[topPiece][pid][topRot][rot];
                        if (v >= 1e8) { valid = false; }
                        else score += v;
                    }
                    
                    if (!valid) continue;

                    SearchState newState = state;
                    newState.grid[nextRow][nextCol] = pid;
                    newState.rotations[pid] = rot;
                    newState.used[pid] = true;
                    newState.score = state.score + score;
                    newState.filledCount = step + 1;
                    nextBeam.push(newState);
                }
            }
        }

        currentBeam.clear();
        while (!nextBeam.empty() && currentBeam.size() < BEAM_WIDTH) {
            currentBeam.push_back(nextBeam.top());
            nextBeam.pop();
        }

        if (currentBeam.empty()) {
            cout << "  Beam died at step " << step << endl;
            break;
        }

        if (step == N - 1 && currentBeam[0].score < bestComplete.score) {
            bestComplete = currentBeam[0];
        }
    }

    if (bestComplete.filledCount != N) {
        cerr << "No complete solution found" << endl;
        return emptyLayout;
    }

    PuzzleLayout result;
    result.rows = rows;
    result.cols = cols;
    result.grid = bestComplete.grid;

    vector<int> allWidths, allHeights;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int piece = bestComplete.grid[r][c];
            Size sz = features[piece].img.size();
            allWidths.push_back(sz.width);
            allHeights.push_back(sz.height);
        }
    }
    sort(allWidths.begin(), allWidths.end());
    sort(allHeights.begin(), allHeights.end());
    float medianW = allWidths[allWidths.size()/2];
    float medianH = allHeights[allHeights.size()/2];

    vector<float> rowHeights(rows, medianH);
    vector<float> colWidths(cols, medianW);

    vector<float> yOff(rows + 1, 0);
    vector<float> xOff(cols + 1, 0);

    for (int r = 0; r < rows; r++) {
        float maxHeight = 0;
        for (int c = 0; c < cols; c++) {
            int piece = bestComplete.grid[r][c];
            Size sz = features[piece].img.size();
            maxHeight = max(maxHeight, (float)sz.height);
        }
        yOff[r + 1] = yOff[r] + maxHeight - 1;
    }

    for (int c = 0; c < cols; c++) {
        float maxWidth = 0;
        for (int r = 0; r < rows; r++) {
            int piece = bestComplete.grid[r][c];
            Size sz = features[piece].img.size();
            maxWidth = max(maxWidth, (float)sz.width);
        }
        xOff[c + 1] = xOff[c] + maxWidth - 1;
    }

    // Assign positions
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int piece = bestComplete.grid[r][c];
            int rot = bestComplete.rotations[piece];
            
            Size cellSize(
                static_cast<int>(xOff[c + 1] - xOff[c]),
                static_cast<int>(yOff[r + 1] - yOff[r])
            );
            
            result.positions[piece] = {
                Point2f(xOff[c], yOff[r]),
                rot * 90.0f,
                cellSize
            };
        }
    }

    result.bounds = Rect2f(0, 0, xOff[cols], yOff[rows]);

    cout << "\n=== GRID DEBUG ===" << endl;
    cout << "Column offsets: ";
    for (int c = 0; c <= cols; c++) cout << xOff[c] << " ";
    cout << endl;
    cout << "Column widths: ";
    for (int c = 0; c < cols; c++) cout << (xOff[c+1] - xOff[c]) << " ";
    cout << endl;

    cout << "Row offsets: ";
    for (int r = 0; r <= rows; r++) cout << yOff[r] << " ";
    cout << endl;
    cout << "Row heights: ";
    for (int r = 0; r < rows; r++) cout << (yOff[r+1] - yOff[r]) << " ";
    cout << endl;

    cout << "\nPiece sizes in grid:" << endl;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int piece = bestComplete.grid[r][c];
            Size sz = features[piece].img.size();
            cout << sz.width << "x" << sz.height << " ";
        }
        cout << endl;
    }

    cout << "\n=== SOLUTION ===" << endl;
    for (int r = 0; r < rows; r++) {
        cout << "  ";
        for (int c = 0; c < cols; c++) {
            int p = bestComplete.grid[r][c];
            cout << p << "(" << bestComplete.rotations[p]*90 << ") ";
        }
        cout << endl;
    }
    cout << "Total MGC score: " << bestComplete.score << endl;

    return result;
}

PuzzleLayout solveWithSteps(const vector<PieceFeature>& features, int canvasW, int canvasH,
                            vector<SolvingStep>& steps) {
    int N = static_cast<int>(features.size());
    PuzzleLayout emptyLayout;
    steps.clear();
    
    if (N == 0) return emptyLayout;

    cout << "\n=== MGC PUZZLE SOLVER (with steps) ===" << endl;
    cout << "Pieces: " << N << endl;

    auto gridShape = inferGridShape(features, N, canvasW, canvasH);
    int rows = gridShape.first;
    int cols = gridShape.second;
    cout << "Grid: " << rows << "x" << cols << endl;

    cout << "Building MGC edge caches..." << endl;
    EdgeCache hCache, vCache;
    buildMGCCache(features, hCache, vCache);

    const int BEAM_WIDTH = 5000;
    cout << "Beam search (width=" << BEAM_WIDTH << ")..." << endl;

    priority_queue<SearchState> beam;
    
    for (int seed = 0; seed < N; seed++) {
        SearchState state;
        state.grid.assign(rows, vector<int>(cols, -1));
        state.rotations.assign(N, 0);
        state.used.assign(N, false);
        state.grid[0][0] = seed;
        state.rotations[seed] = 0;
        state.used[seed] = true;
        state.score = 0.0;
        state.filledCount = 1;
        beam.push(state);
    }

    vector<SearchState> currentBeam;
    while (!beam.empty() && currentBeam.size() < BEAM_WIDTH) {
        currentBeam.push_back(beam.top());
        beam.pop();
    }

    if (!currentBeam.empty()) {
        SolvingStep initialStep;
        initialStep.grid = currentBeam[0].grid;
        for (int i = 0; i < N; i++) {
            initialStep.rotations[i] = currentBeam[0].rotations[i];
        }
        initialStep.stepNumber = 0;
        initialStep.filledCount = currentBeam[0].filledCount;
        initialStep.score = currentBeam[0].score;
        steps.push_back(initialStep);
    }

    SearchState bestComplete;
    bestComplete.filledCount = 0;
    bestComplete.score = 1e9;

    for (int step = 1; step < N; step++) {
        int nextRow = step / cols;
        int nextCol = step % cols;
        
        if (step % 5 == 0) {
            cout << "  Step " << step << "/" << N 
                 << ", beam size: " << currentBeam.size() 
                 << ", best score: " << currentBeam[0].score << endl;
        }

        priority_queue<SearchState> nextBeam;

        for (const auto& state : currentBeam) {
            for (int pid = 0; pid < N; pid++) {
                if (state.used[pid]) continue;

                for (int rot = 0; rot < 4; rot++) {
                    double score = 0.0;
                    bool valid = true;
                    
                    if (nextCol > 0) {
                        int leftPiece = state.grid[nextRow][nextCol - 1];
                        int leftRot = state.rotations[leftPiece];
                        double h = hCache[leftPiece][pid][leftRot][rot];
                        if (h >= 1e8) { valid = false; }
                        else score += h;
                    }
                    
                    if (valid && nextRow > 0) {
                        int topPiece = state.grid[nextRow - 1][nextCol];
                        int topRot = state.rotations[topPiece];
                        double v = vCache[topPiece][pid][topRot][rot];
                        if (v >= 1e8) { valid = false; }
                        else score += v;
                    }
                    
                    if (!valid) continue;

                    SearchState newState = state;
                    newState.grid[nextRow][nextCol] = pid;
                    newState.rotations[pid] = rot;
                    newState.used[pid] = true;
                    newState.score = state.score + score;
                    newState.filledCount = step + 1;
                    nextBeam.push(newState);
                }
            }
        }

        currentBeam.clear();
        while (!nextBeam.empty() && currentBeam.size() < BEAM_WIDTH) {
            currentBeam.push_back(nextBeam.top());
            nextBeam.pop();
        }

        if (currentBeam.empty()) {
            cout << "  Beam died at step " << step << endl;
            break;
        }

        if (!currentBeam.empty()) {
            SolvingStep stepInfo;
            stepInfo.grid = currentBeam[0].grid;
            for (int i = 0; i < N; i++) {
                stepInfo.rotations[i] = currentBeam[0].rotations[i];
            }
            stepInfo.stepNumber = step;
            stepInfo.filledCount = currentBeam[0].filledCount;
            stepInfo.score = currentBeam[0].score;
            steps.push_back(stepInfo);
        }

        if (step == N - 1 && currentBeam[0].score < bestComplete.score) {
            bestComplete = currentBeam[0];
        }
    }

    if (bestComplete.filledCount != N) {
        cerr << "No complete solution found" << endl;
        return emptyLayout;
    }

    PuzzleLayout result;
    result.rows = rows;
    result.cols = cols;
    result.grid = bestComplete.grid;

    vector<int> allWidths, allHeights;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int piece = bestComplete.grid[r][c];
            Size sz = features[piece].img.size();
            allWidths.push_back(sz.width);
            allHeights.push_back(sz.height);
        }
    }
    sort(allWidths.begin(), allWidths.end());
    sort(allHeights.begin(), allHeights.end());
    float medianW = allWidths[allWidths.size()/2];
    float medianH = allHeights[allHeights.size()/2];

    vector<float> rowHeights(rows, medianH);
    vector<float> colWidths(cols, medianW);

    vector<float> yOff(rows + 1, 0);
    vector<float> xOff(cols + 1, 0);

     const int OVERLAP = 2;  // 2 pixels of overlap between adjacent pieces

    for (int r = 0; r < rows; r++) {
        float maxHeight = 0;
        for (int c = 0; c < cols; c++) {
            int piece = bestComplete.grid[r][c];
            Size sz = features[piece].img.size();
            maxHeight = max(maxHeight, (float)sz.height);
        }
        yOff[r + 1] = yOff[r] + maxHeight - 1;
        // yOff[r + 1] = yOff[r] + maxHeight - OVERLAP;
    }

    for (int c = 0; c < cols; c++) {
        float maxWidth = 0;
        for (int r = 0; r < rows; r++) {
            int piece = bestComplete.grid[r][c];
            Size sz = features[piece].img.size();
            maxWidth = max(maxWidth, (float)sz.width);
        }
        xOff[c + 1] = xOff[c] + maxWidth - 1;
        // xOff[c + 1] = xOff[c] + maxWidth - OVERLAP;
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int piece = bestComplete.grid[r][c];
            int rot = bestComplete.rotations[piece];
            
            Size cellSize(
                static_cast<int>(xOff[c + 1] - xOff[c]),
                static_cast<int>(yOff[r + 1] - yOff[r])
            );
            
            result.positions[piece] = {
                Point2f(xOff[c], yOff[r]),
                rot * 90.0f,
                cellSize
            };
        }
    }

    result.bounds = Rect2f(0, 0, xOff[cols], yOff[rows]);

    cout << "\n=== SOLUTION ===" << endl;
    for (int r = 0; r < rows; r++) {
        cout << "  ";
        for (int c = 0; c < cols; c++) {
            int p = bestComplete.grid[r][c];
            cout << p << "(" << bestComplete.rotations[p]*90 << ") ";
        }
        cout << endl;
    }
    cout << "Total MGC score: " << bestComplete.score << endl;
    cout << "Recorded " << steps.size() << " solving steps" << endl;

    return result;
}


}  // namespace PieceMatcher