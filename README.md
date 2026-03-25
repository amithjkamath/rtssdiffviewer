---
title: RTSS Diff Viewer
emoji: CT
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: false
---

# RTSS Diff Viewer

A standalone Streamlit app for:

- converting RTSS `.dcm` files to JSON
- downloading normalized JSON outputs
- visualizing intelligent diffs between two RTSS versions (slice-by-slice for contours)
- batch-uploading multiple variants and switching any two versions for comparison
- efficiently handling large RTSS files with many contour points

## Component Comparison Modes

The app supports comparing four different components:

- **metadata**: DICOM package-level tags using standard unified text diff (fast)
- **structures**: ROI structure definitions using text diff
- **references**: Reference frame sequences using text diff  
- **contours**: Point-cloud comparison grouped by slice plane (z-coordinate)
  - Optimized for large files with many contour points (>50)
  - Two-panel left-right layout for easy visual comparison
  - Shows points organized by slice rather than full text diff
  - Clearly marks which ROIs/slices are identical, different, or missing
  - Much faster and more actionable for clinical workflows

## Contour Comparison Features

When comparing contour data:
- **Slice-by-slice organization**: Each z-coordinate (slice plane) shown in a collapsible expander
- **Two-panel layout**: Left file contents on the left, right file contents on the right
- **Visual indicators**:
  - ✓ Identical ROI and points
  - ⚠️ Different points in ROI  
  - ⊘ ROI only exists in one file
  - ✅ Entire slice is identical
  - ❌ No contour data on this slice
- **Component equality**: When metadata, structures, or references are identical, a clear message indicates no differences

## User Workflows

- `Instructions` mode: in-app overview of Pair Mode and Batch Compare usage.
- `Pair Mode`: upload two RTSS files, inspect structured diff by component, and view contour points in 3D.
- `Batch Compare`: upload multiple RTSS files and compare any two using your choice of comparison component.

## Developer Setup

```bash
make install
make run
```

## Developer Testing

```bash
make test
```

To install dependencies and run tests in one step:

```bash
make test-all
```

## Hugging Face Spaces Deployment

Use the deploy workflow in this repository:

```bash
make deploy-init SPACE=amithjkamath/rtssdiffviewer
make deploy
```

This pushes local `deploy` branch to Space `main`.
