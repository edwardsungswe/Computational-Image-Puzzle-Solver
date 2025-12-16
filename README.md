# Computational Image Puzzle Solver

This repository contains a **Computational Image Puzzle Solver** that reconstructs a larger image from scattered puzzle pieces using computer vision and image processing techniques. It was developed as a project for CSCI 576.

## Overview

The goal of this project is to automatically identify, match, and assemble image fragments (puzzle pieces) into a complete image. The solver loads input images containing scattered pieces, detects and extracts individual pieces, compares visual and edge features, and computes an arrangement that best reconstructs the original image.

## Features

- Detects and extracts puzzle pieces from input images.
- Analyzes visual features and edge shapes of pieces for matching.
- Searches for optimal placement and orientation of pieces.

## Requirements

- C++17 or newer compiler
- CMake (>= 3.10)
- OpenCV (version 3.x or 4.x)
