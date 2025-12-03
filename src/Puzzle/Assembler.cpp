#include "Assembler.h"

using namespace cv;
using namespace std;

namespace Assembler {

vector<Point> computePiecePositions(
    const PuzzleLayout& layout,
    const vector<PieceFeature>& features,
    int canvasW,
    int canvasH)
{
    int R = layout.grid.size();
    int C = layout.grid[0].size();

    vector<int> rowHeights(R, 0), colWidths(C, 0);

    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (layout.grid[r][c] >= 0)
                rowHeights[r] = max(rowHeights[r],
                                    features[layout.grid[r][c]].img.rows);

    for (int c = 0; c < C; c++)
        for (int r = 0; r < R; r++)
            if (layout.grid[r][c] >= 0)
                colWidths[c] = max(colWidths[c],
                                   features[layout.grid[r][c]].img.cols);

    vector<int> xOffset(C, 0), yOffset(R, 0);
    for (int c = 1; c < C; c++) xOffset[c] = xOffset[c-1] + colWidths[c-1];
    for (int r = 1; r < R; r++) yOffset[r] = yOffset[r-1] + rowHeights[r-1];

    int compactW = xOffset[C-1] + colWidths[C-1];
    int compactH = yOffset[R-1] + rowHeights[R-1];

    int baseX = (canvasW - compactW) / 2;
    int baseY = (canvasH - compactH) / 2;

    vector<Point> pos(features.size());
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            int idx = layout.grid[r][c];
            if (idx < 0) continue;
            pos[idx] = Point(baseX + xOffset[c], baseY + yOffset[r]);
        }
    }

    return pos;
}


Mat assembleImage(const PuzzleLayout& layout,
                  const vector<PieceFeature>& features,
                  int canvasW,
                  int canvasH)
{
    vector<Point> pos = computePiecePositions(layout, features, canvasW, canvasH);

    Mat canvas(canvasH, canvasW, CV_8UC3, Scalar(0,0,0));

    for (int i = 0; i < features.size(); i++) {
        const Mat& piece = features[i].img;
        const Point& p = pos[i];

        if (p.x >= 0 && p.y >= 0 &&
            p.x + piece.cols <= canvasW &&
            p.y + piece.rows <= canvasH)
        {
            piece.copyTo(canvas(Rect(p.x, p.y, piece.cols, piece.rows)));
        }
    }

    return canvas;
}

}