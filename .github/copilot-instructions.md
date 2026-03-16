# Copilot Instructions for `sdc-c2-fusion-starter`

This repository is a Python-based sensor fusion and tracking starter project built around the Waymo Open Dataset. Prefer repository-specific guidance over generic autonomous driving patterns.

## Repository focus

- The main workflow lives in `loop_over_dataset.py`.
- Object detection code is split across `student/objdet_pcl.py`, `student/objdet_detect.py`, and `student/objdet_eval.py`.
- Tracking code is split across `student/filter.py`, `student/association.py`, `student/measurements.py`, and `student/trackmanagement.py`.
- Reusable helpers live in `misc/`.
- Third-party or external model code lives in `tools/`; avoid changing it unless the task explicitly requires it.

## How to work in this repo

- Prefer making changes in `student/` first. This is where the intended exercise and implementation points live.
- Keep the current procedural style unless there is a strong reason to refactor. Do not introduce large frameworks or architecture changes.
- Preserve the existing execution flow in `loop_over_dataset.py`, especially the `exec_detection`, `exec_tracking`, and `exec_visualization` toggles.
- Preserve the fallback behavior that loads precomputed artifacts from `results/` when a step is not selected for execution.
- Respect the current import strategy and path handling used throughout the project.

## Coding guidance

- This codebase is Python-first. Prioritize Python, NumPy, OpenCV, PyTorch, and the local Waymo reader over ROS 2 or C++ suggestions unless the task explicitly asks for them.
- Follow the existing numeric style. Much of the tracking code uses `numpy.matrix`, explicit state vectors, covariance matrices, and calibration transforms.
- For perception and tracking changes, be explicit about coordinate frames, dimensions, and sensor-to-vehicle transforms.
- Prefer deterministic logic for tracking and evaluation. Avoid introducing nondeterministic behavior into core processing loops.
- Handle edge cases explicitly, especially invalid measurements, out-of-range objects, field-of-view checks, and missing model files.
- Keep comments sparse and practical. Add them only when the math, transformation logic, or tracking behavior is not obvious from the code.

## File-specific expectations

- In `student/` files, preserve existing task markers and surrounding structure unless the task is to remove or replace them.
- In `student/objdet_detect.py`, keep model configuration, device selection, and decoding/post-processing logic aligned with the selected architecture (`darknet` or `fpn_resnet`).
- In tracking modules, preserve the Kalman filter and track-management flow: predict, associate, update, manage tracks.
- In visualization-related code, avoid changing output behavior unless the task is specifically about display or debugging.

## Dependencies and runtime assumptions

- This project expects dataset files under `dataset/`.
- Pretrained model weights are expected in the `tools/objdet_models/.../pretrained` directories.
- The lightweight Waymo reader under `tools/waymo_reader/` is part of the intended local setup.
- Prefer solutions that work with the existing Python environment and `requirements.txt`.

## Interaction preferences from `.github/AGENT.md`

- When proposing or implementing perception logic, explain the mathematical or geometric reasoning before or alongside the code when helpful.
- Prioritize low-overhead, practical solutions.
- Call out risks to determinism, latency, or safety-relevant behavior when they are relevant to the requested change.
