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
- visualizing git-like diffs between two RTSS versions
- batch-uploading multiple variants and switching any two versions for comparison

## Local Setup

```bash
make install
make run
```

## Hugging Face Spaces Deployment

Use the deploy workflow in this repository:

```bash
make deploy-init SPACE=amithjkamath/rtssdiffviewer
make deploy
```

This pushes local `deploy` branch to Space `main`.
