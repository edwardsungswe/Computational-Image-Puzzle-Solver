#include "PieceMatcher_translate.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <iostream>

using namespace std;
using namespace cv;

/*** --- Edge Distance (translation-only) --- ***/
static double edgeLumaDist(const EdgeFeature& A, const EdgeFeature& B)
{
    double s = 0.0;
    for (int i = 0; i < A.vals.size(); i++)
    {
        double d = A.vals[i] - B.vals[i];
        s += d * d;
    }
    return s;
}

/*** Compute match distance for complementary edges only ***/
static double edgeDistanceTranslate(const PieceFeature& A, const PieceFeature& B, int eA, int eB)
{
    const EdgeFeature* EA = nullptr;
    const EdgeFeature* EB = nullptr;

    switch(eA){
        case 0: EA = &A.top; break;
        case 1: EA = &A.right; break;
        case 2: EA = &A.bottom; break;
        case 3: EA = &A.left; break;
    }
    switch(eB){
        case 0: EB = &B.top; break;
        case 1: EB = &B.right; break;
        case 2: EB = &B.bottom; break;
        case 3: EB = &B.left; break;
    }

    return edgeLumaDist(*EA, *EB);
}

/*** Helper to test complementary relation (translation-only) ***/
static inline bool complementaryEdges(int eA, int eB)
{
    return (eA == 0 && eB == 2) ||  // top-bottom
           (eA == 2 && eB == 0) ||
           (eA == 1 && eB == 3) ||  // right-left
           (eA == 3 && eB == 1);
}

namespace PieceMatcher_translate {

vector<Pair> createFilteredMatches(
        const vector<PieceFeature>& features,
        double ratioTestThreshold)
{
    vector<Pair> matches;
    int N = features.size();

    for(int i = 0; i < N; i++)
    {
        for(int j = i+1; j < N; j++)
        {
            for(int eA = 0; eA < 4; eA++)
            {
                for(int eB = 0; eB < 4; eB++)
                {
                    if(!complementaryEdges(eA, eB))
                        continue;

                    double score = edgeDistanceTranslate(features[i], features[j], eA, eB);
                    matches.push_back({i, j, eA, eB, score});
                }
            }
        }
    }

    sort(matches.begin(), matches.end(), [](auto &a, auto &b){
        return a.val < b.val;
    });

    const int K = min(20, (int)matches.size());
    double avg = 0.0;
    for(int i = 0; i < K; i++) avg += matches[i].val;
    avg /= K;

    double cutoff = avg * 5.0;

    vector<Pair> filtered;
    unordered_map<int,double> bestScore;

    for(auto &m : matches)
    {
        if(m.val > cutoff) continue;

        int keyA = m.pieceA * 10 + m.edgeA;
        int keyB = m.pieceB * 10 + m.edgeB;

        bool ok = true;

        if(bestScore.count(keyA))
        {
            if(bestScore[keyA] / m.val > ratioTestThreshold)
                ok = false;
        }
        else bestScore[keyA] = m.val;

        if(bestScore.count(keyB))
        {
            if(bestScore[keyB] / m.val > ratioTestThreshold)
                ok = false;
        }
        else bestScore[keyB] = m.val;

        if(ok) filtered.push_back(m);
    }

    return filtered;
}


/*** Build layout without rotation ***/
PuzzleLayout buildLayout(
    const vector<Pair>& matches,
    const vector<PieceFeature>& f,
    int canvasW,
    int canvasH)
{
PuzzleLayout layout;
int N = f.size();

unordered_map<int,PiecePosition> pos;
vector<bool> placed(N,false);

// Build adjacency list (translation-only)
vector<vector<pair<int,int>>> adj(N);  
// adj[A] = { (B, edgeA) }

for(const auto& m : matches)
{
    adj[m.pieceA].push_back({m.pieceB, m.edgeA});
    adj[m.pieceB].push_back({m.pieceA, m.edgeB});
}

auto placeNeighbor = [&](int a, int b, int eA){
    Size sa = f[a].img.size();
    Size sb = f[b].img.size();
    Point2f base = pos[a].position;

    Point2f ofs;
    if(eA == 0) ofs = Point2f(0,-sb.height);
    else if(eA == 2) ofs = Point2f(0, sa.height);
    else if(eA == 1) ofs = Point2f(sa.width,0);
    else ofs = Point2f(-sb.width,0);

    pos[b] = { base + ofs, 0.0f, sb };
    placed[b] = true;
};

queue<int> q;

// We must run BFS for all connected components
for(int root = 0; root < N; root++)
{
    if(placed[root]) continue;

    // Start new group: place this root at (0,0)
    pos[root] = { Point2f(0,0), 0.0f, f[root].img.size() };
    placed[root] = true;
    q.push(root);

    while(!q.empty())
    {
        int a = q.front(); q.pop();

        for(auto& nb : adj[a])
        {
            int b = nb.first;
            int eA = nb.second;

            if(!placed[b])
            {
                placeNeighbor(a,b,eA);
                q.push(b);
            }
        }
    }
}

// Normalize to positive coordinates
float x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
for(auto& kv : pos)
{
    Point2f p = kv.second.position;
    Size s = kv.second.size;
    x0 = min(x0,p.x);
    y0 = min(y0,p.y);
    x1 = max(x1,p.x+s.width);
    y1 = max(y1,p.y+s.height);
}

float shiftX = -x0 + 50;
float shiftY = -y0 + 50;

for(auto& kv : pos)
{
    kv.second.position.x += shiftX;
    kv.second.position.y += shiftY;
}

layout.positions = pos;
layout.bounds = Rect2f(0,0, x1-x0, y1-y0);
return layout;
}

}
