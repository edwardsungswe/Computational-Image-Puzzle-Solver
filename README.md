# Computational Image Puzzle Solver

A computer vision application that automatically reconstructs images from scattered puzzle pieces using gradient-based edge matching and beam search optimization.

## Overview

This project solves the jigsaw puzzle assembly problem: given a set of scattered image pieces, automatically determine their correct positions and orientations to reconstruct the original image. The solver uses the **Mahalanobis Gradient Compatibility (MGC)** metric to evaluate edge compatibility between pieces and employs **beam search** to efficiently explore the solution space.

## Features

- Automatic piece detection and extraction from input images
- Rotation-invariant piece matching (supports 0°, 90°, 180°, 270° rotations)
- Statistical edge compatibility using gradient analysis
- Beam search optimization for efficient puzzle solving
- Animated visualization of the solving process
- Seamless piece assembly with edge blending

## Pipeline

```
Input Image → Piece Extraction → Feature Extraction → Piece Matching → Assembly → Output
```

### 1. Piece Extraction

The `PieceExtractor` module detects and isolates individual puzzle pieces:

1. **Grayscale Conversion & Thresholding**: Convert to grayscale and create a binary mask using intensity thresholding
2. **Morphological Operations**: Apply closing and opening operations to clean the mask and remove noise
3. **Contour Detection**: Find external contours to identify individual pieces
4. **Rotation Normalization**: Use `minAreaRect` to detect piece orientation and de-rotate pieces to axis-aligned bounding boxes
5. **Artifact Cleaning**: Remove dark border artifacts using inpainting and edge cropping

### 2. Feature Extraction

The `FeatureExtractor` module computes edge features for matching:

- **Sobel Gradient Computation**: Calculate horizontal (`Gx`) and vertical (`Gy`) gradients using 3x3 Sobel operators
- **Boundary Pixel Extraction**: Extract pixel values along each of the four edges (top, right, bottom, left)
- **Gradient Profiles**: Store gradient values along edges for texture-aware matching

### 3. Piece Matching (Core Algorithm)

The matching algorithm uses **Mahalanobis Gradient Compatibility (MGC)**, based on techniques from Gallagher's work on jigsaw puzzle solving.

#### Mahalanobis Gradient Compatibility (MGC)

MGC measures how well two puzzle piece edges fit together by comparing their gradient distributions statistically.

**Key Insight**: When two pieces are correctly adjacent, the gradient pattern across their boundary should be consistent with the gradient patterns within each piece.

##### Algorithm Steps

For two pieces A (left) and B (right) being tested for horizontal adjacency:

1. **Compute Internal Gradients**

   For each row along the edges:

   ```
   G_L[i] = (R, G, B) gradient at right edge of piece A
          = pixel[last] - pixel[last-1] for each channel

   G_R[i] = (R, G, B) gradient at left edge of piece B
          = pixel[1] - pixel[0] for each channel
   ```

2. **Compute Boundary Gradient**

   ```
   G_LR[i] = (R, G, B) gradient across the boundary
           = B.pixel[0] - A.pixel[last] for each channel
   ```

3. **Add Dummy Gradients**

   For numerical stability, add a set of dummy gradient samples (from Gallagher's paper):

   ```
   Dummies = {(0,0,0), (1,1,1), (-1,-1,-1), (0,0,1), (0,1,0), (1,0,0), (-1,0,0), (0,-1,0), (0,0,-1)}
   ```

4. **Compute Statistics**

   Calculate mean (μ) and covariance matrix (Σ) for G_L and G_R:

   ```
   μ = (1/n) Σ samples
   Σ = (1/(n-1)) Σ (sample - μ)(sample - μ)^T + λI
   ```

   where λ is a regularization term for numerical stability.

5. **Mahalanobis Distance**

   Measure how well the boundary gradients fit each piece's gradient distribution:

   ```
   D_L = Σ mahalanobis(G_LR[i], μ_L, Σ_L^(-1))
   D_R = Σ mahalanobis(G_LR[i], μ_R, Σ_R^(-1))
   ```

   The Mahalanobis distance is computed as:

   ```
   d(x, μ, Σ^(-1)) = sqrt((x - μ)^T Σ^(-1) (x - μ))
   ```

6. **Compatibility Score**
   ```
   MGC(A, B) = min(D_L, D_R)
   ```
   Lower scores indicate better compatibility.

##### Why Mahalanobis Distance?

Unlike Euclidean distance, Mahalanobis distance:

- **Accounts for correlation**: RGB channels are often correlated; Mahalanobis considers this
- **Scale-invariant**: Normalizes by variance, so high-variance edges don't dominate
- **Distribution-aware**: Measures how "typical" a gradient is for a given edge

##### Size Compatibility Check

Before computing MGC, pieces are checked for size compatibility:

```
ratio = height_A / height_B
if (ratio < 0.8 || ratio > 1.25): return INFINITY  // Incompatible
```

#### Beam Search Optimization

Finding the optimal arrangement of N pieces in an R×C grid is computationally expensive. The solver uses **beam search** to efficiently explore the solution space.

**Algorithm:**

1. **Initialization**: Start with each piece as a potential first piece (position [0,0])

2. **Pre-computation**: Build MGC caches for all piece pairs and rotations:

   ```
   hCache[a][b][rot_a][rot_b] = MGC score for A-left-of-B
   vCache[a][b][rot_a][rot_b] = MGC score for A-above-B
   ```

3. **Iterative Expansion**: For each grid position (row-major order):

   - For each state in the current beam:
     - Try placing each unused piece in each rotation (0°, 90°, 180°, 270°)
     - Compute compatibility with left neighbor (if exists) and top neighbor (if exists)
     - Add rotation penalty to prefer non-rotated solutions when scores are similar
   - Keep top `BEAM_WIDTH` (default: 5000) states by cumulative score

4. **Solution Selection**: Return the complete state with the lowest total score

**Complexity**: O(N² × 4² × BEAM_WIDTH) where N is the number of pieces

#### Grid Shape Inference

The solver automatically infers the grid dimensions by:

1. Computing average piece dimensions
2. Testing all factor pairs of N (number of pieces)
3. Selecting the grid shape whose aspect ratio best matches the canvas aspect ratio

### 4. Assembly

The `Assembler` module places pieces according to the computed layout:

1. **Position Calculation**: Compute (x, y) offsets for each grid cell based on piece sizes
2. **Rotation Application**: Rotate pieces to their solved orientations
3. **Overlap Handling**: For rotated puzzles, apply slight overlap to eliminate interpolation gaps
4. **Edge Blending**: Use weighted averaging at piece boundaries for seamless transitions

## Requirements

- C++17 or newer compiler
- CMake (>= 3.10)
- OpenCV (version 3.x or 4.x)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./puzzle_solver <input_image> [output_directory]
```

The solver will:

1. Load the input image containing scattered pieces
2. Extract and process individual pieces
3. Solve the puzzle using MGC-based matching
4. Output the assembled image and solving animation

## Project Structure

```
src/
├── Main.cpp                    # Entry point
├── Pieces/
│   ├── PieceExtractor.h/cpp    # Piece detection and extraction
├── Features/
│   ├── FeatureExtractor.h/cpp  # Edge feature computation
├── Matching/
│   ├── PieceMatcher.h/cpp      # MGC algorithm and beam search
├── Puzzle/
│   └── Assembler.h/cpp         # Final image assembly
└── Animation/
    └── PuzzleAnimator.h/cpp    # Solving visualization
```

## Algorithm Parameters

| Parameter          | Default | Description                                  |
| ------------------ | ------- | -------------------------------------------- |
| `BEAM_WIDTH`       | 5000    | Number of states to keep at each search step |
| `ROTATION_PENALTY` | 50.0    | Score penalty for non-zero rotations         |
| `MIN_AREA`         | 100     | Minimum contour area for piece detection     |
| `EDGE_SKIP`        | 1       | Pixels to skip from edge to avoid artifacts  |
| `REGULARIZATION`   | 1.0     | Covariance matrix regularization term        |

## References

- Gallagher, A. C. (2012). "Jigsaw puzzles with pieces of unknown orientation." CVPR.
- Pomeranz, D., Shemesh, M., & Ben-Shahar, O. (2011). "A fully automated greedy square jigsaw puzzle solver." CVPR.
