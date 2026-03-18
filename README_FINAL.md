# Replicating the Final Project Results

> **Udacity SDC Nanodegree — Sensor Fusion and Object Tracking**  
> Step-by-step instructions to reproduce all four tracking scenarios and the tracking movie.

---

## Prerequisites

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install easydict  # required by fpn_resnet model loader
```

### 2. Verify dataset files

The following `.tfrecord` files must be present in `dataset/`:

```
dataset/
  training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord   # Seq 1
  training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord  # Seq 2
```

### 3. Verify pre-computed detections

Pre-computed lidar detections must be present in:

```
results/Lidar_Detections_Tracking_Final_Project/
```

This folder contains `.pkl` files for both sequences and is used by all four steps (detection is skipped, results are loaded from file).

---

## Running Each Step

All four steps share the same entry point: `loop_over_dataset.py`. Only a few lines need to change between steps.

> **Note:** The current `loop_over_dataset.py` is configured for **Step 4** (the final scenario). To reproduce earlier steps, apply the changes described below.

---

### Step 1 — Extended Kalman Filter

**Scenario:** Sequence 2, frames 150–200, single target.

Edit `loop_over_dataset.py`:

```python
# Line ~53-56
data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord'  # Seq 2
show_only_frames = [150, 200]

# Line ~69
configs_det.lim_y = [-5, 10]
```

Run:

```bash
python loop_over_dataset.py
```

**Expected result:** Mean RMSE ≤ 0.35 for a single track (achieved: **0.23**).

---

### Step 2 — Track Management

**Scenario:** Sequence 2, frames 65–100, single target appears then disappears.

Edit `loop_over_dataset.py`:

```python
data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord'  # Seq 2
show_only_frames = [65, 100]
configs_det.lim_y = [-5, 15]
```

Run:

```bash
python loop_over_dataset.py
```

**Expected result:** Console shows `deleting track no. 0`. One clean track line in the RMSE plot (achieved mean RMSE: **0.61** — expected due to systematic lidar y-offset).

---

### Step 3 — Data Association (lidar only)

**Scenario:** Sequence 1, frames 0–200, multiple targets.

Edit `loop_over_dataset.py`:

```python
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'  # Seq 1
show_only_frames = [0, 200]
configs_det.lim_y = [-25, 25]
```

Run:

```bash
python loop_over_dataset.py
```

**Expected result:** Multiple confirmed tracks, no confirmed ghost tracks. Achieved RMSE: **0.15 / 0.12 / 0.19** for tracks 0, 1, 10.

---

### Step 4 — Camera-Lidar Sensor Fusion *(default configuration)*

**Scenario:** Sequence 1, frames 0–200, multiple targets with both lidar and camera.

This is the **current default** in `loop_over_dataset.py`. Simply run:

```bash
python loop_over_dataset.py
```

**Expected result:**
- Console shows lidar updates followed by camera updates for each frame
- 3 confirmed tracks (0, 1, 10), no track losses over 200 frames
- Mean RMSE: **0.17 / 0.10 / 0.12** (all ≤ 0.25 target)

---

### Generating the Tracking Movie

To produce `my_tracking_results.avi` (Step 4 configuration required):

1. In `loop_over_dataset.py`, change:
   ```python
   exec_visualization = ['show_tracks', 'make_tracking_movie']
   ```

2. Run:
   ```bash
   python loop_over_dataset.py
   ```

3. The movie is saved to:
   ```
   results/Lidar_Detections_Tracking_Final_Project/my_tracking_results.avi
   ```
   Move it to the `media/` folder so it is tracked by git (the `results/` folder is in `.gitignore`):
   ```bash
   mv results/Lidar_Detections_Tracking_Final_Project/my_tracking_results.avi media/
   ```

4. Restore to `['show_tracks']` after generating.

> [!NOTE] 
> Running headlessly (no display) requires setting `MPLBACKEND=Agg`:
> ```bash
> MPLBACKEND=Agg python loop_over_dataset.py
> ```

---

## Implemented Files

| File | What was implemented |
|---|---|
| [`student/filter.py`](student/filter.py) | `F()`, `Q()`, `predict()`, `update()`, `gamma()`, `S()` |
| [`student/trackmanagement.py`](student/trackmanagement.py) | `Track.__init__()`, `manage_tracks()`, `handle_updated_track()` |
| [`student/association.py`](student/association.py) | `associate()`, `get_closest_track_and_meas()`, `gating()`, `MHD()` |
| [`student/measurements.py`](student/measurements.py) | `Sensor.in_fov()`, `Sensor.get_hx()`, `Sensor.generate_measurement()`, `Measurement.__init__()` for camera |

### Critical implementation note — camera projection signs

The nonlinear camera projection in `get_hx()` uses **negative signs** to be consistent with the Jacobian in `get_H()`:

```python
u = self.c_i - self.f_i * y_cam / x_cam
v = self.c_j - self.f_j * z_cam / x_cam
```

Using positive signs (the "intuitive" formula) causes ~100px projection error, making all Mahalanobis distances too large to pass chi-squared gating — camera measurements never associate and tracks never get camera updates.

---

## Writeup

See [`writeup-final.md`](writeup-final.md) for:
- Short recap of all four steps and results
- Benefits of camera-lidar fusion vs. lidar-only
- Real-life challenges for sensor fusion systems
- Ideas for future improvements
