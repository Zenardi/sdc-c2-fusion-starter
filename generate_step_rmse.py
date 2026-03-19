"""
Regenerate RMSE plots for individual tracking steps.

Usage:
    python generate_step_rmse.py --step 1   # EKF lidar-only
    python generate_step_rmse.py --step 2   # Track management lidar-only
    python generate_step_rmse.py --step 3   # Association lidar-only
    python generate_step_rmse.py --step 4   # Camera-lidar fusion

Each step uses the correct scenario configuration from the course rubric and
saves the RMSE figure to img/writeup_final/stepN_rmse.png.
"""

import argparse
import copy
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# ── Step configurations ──────────────────────────────────────────────────────
STEP_CONFIGS = {
    1: dict(
        sequence=2,
        data_filename='training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord',
        frames=[150, 200],
        lim_y=[-5, 10],
        sensors=['lidar'],
        save_path='img/writeup_final/step1_rmse.png',
        description='EKF, lidar-only, Seq 2 frames 150-200',
    ),
    2: dict(
        sequence=2,
        data_filename='training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord',
        frames=[65, 100],
        lim_y=[-5, 15],
        sensors=['lidar'],
        save_path='img/writeup_final/step2_rmse.png',
        description='Track management, lidar-only, Seq 2 frames 65-100',
    ),
    3: dict(
        sequence=1,
        data_filename='training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord',
        frames=[0, 200],
        lim_y=[-25, 25],
        sensors=['lidar'],
        save_path='img/writeup_final/step3_rmse.png',
        description='Association, lidar-only, Seq 1 frames 0-200',
    ),
    4: dict(
        sequence=1,
        data_filename='training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord',
        frames=[0, 200],
        lim_y=[-25, 25],
        sensors=['lidar', 'camera'],
        save_path='img/writeup_final/step4_rmse.png',
        description='Camera-lidar fusion, Seq 1 frames 0-200',
    ),
}


def run_step(step_num):
    cfg = STEP_CONFIGS[step_num]
    print(f'\n=== Running Step {step_num}: {cfg["description"]} ===')

    # ── Patch params before any student module loads them ────────────────────
    import misc.params as params
    params.tracking_sensors = cfg['sensors']

    # ── Waymo reader ─────────────────────────────────────────────────────────
    from tools.waymo_reader.simple_waymo_open_dataset_reader import (
        utils as waymo_utils, WaymoDataFileReader, dataset_pb2, label_pb2)
    import misc.objdet_tools as tools
    from misc.helpers import load_object_from_file

    import student.objdet_detect as det
    from student.filter import Filter
    from student.trackmanagement import Trackmanagement
    from student.association import Association
    from student.measurements import Sensor, Measurement
    from misc.evaluation import plot_rmse

    # ── Dataset ───────────────────────────────────────────────────────────────
    base = os.path.dirname(os.path.realpath(__file__))
    data_fullpath = os.path.join(base, 'dataset', cfg['data_filename'])
    results_fullpath = os.path.join(base, 'results', 'Lidar_Detections_Tracking_Final_Project')
    save_path = os.path.join(base, cfg['save_path'])

    datafile = WaymoDataFileReader(data_fullpath)
    datafile_iter = iter(datafile)

    # ── Detection config ──────────────────────────────────────────────────────
    configs_det = det.load_configs(model_name='fpn_resnet')
    configs_det.use_labels_as_objects = False
    configs_det.lim_y = cfg['lim_y']

    # ── Tracking objects ──────────────────────────────────────────────────────
    KF = Filter()
    association = Association()
    manager = Trackmanagement()
    lidar = None
    camera = None
    np.random.seed(10)

    all_labels = []
    cnt_frame = 0
    frame_start, frame_end = cfg['frames']

    while True:
        try:
            frame = next(datafile_iter)
            if cnt_frame < frame_start:
                cnt_frame += 1
                continue
            elif cnt_frame > frame_end:
                print('Reached end of selected frames')
                break

            print(f'  frame {cnt_frame}', end='\r')

            lidar_name = dataset_pb2.LaserName.TOP
            camera_name = dataset_pb2.CameraName.FRONT
            lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)
            camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)

            # ── Load cached detections ────────────────────────────────────────
            detections = load_object_from_file(results_fullpath, cfg['data_filename'], 'detections', cnt_frame)
            valid_label_flags = load_object_from_file(results_fullpath, cfg['data_filename'], 'valid_labels', cnt_frame)

            # ── Init sensors ──────────────────────────────────────────────────
            if lidar is None:
                lidar = Sensor('lidar', lidar_calibration)
            if camera is None:
                camera = Sensor('camera', camera_calibration)

            # ── Lidar measurements ────────────────────────────────────────────
            meas_list_lidar = []
            for detection in detections:
                if (detection[1] > configs_det.lim_x[0] and detection[1] < configs_det.lim_x[1]
                        and detection[2] > configs_det.lim_y[0] and detection[2] < configs_det.lim_y[1]):
                    meas_list_lidar = lidar.generate_measurement(cnt_frame, detection[1:], meas_list_lidar)

            # ── Camera measurements ───────────────────────────────────────────
            meas_list_cam = []
            for label in frame.camera_labels[0].labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    box = label.box
                    z = [box.center_x, box.center_y, box.width, box.length]
                    z[0] += np.random.normal(0, params.sigma_cam_i)
                    z[1] += np.random.normal(0, params.sigma_cam_j)
                    meas_list_cam = camera.generate_measurement(cnt_frame, z, meas_list_cam)

            # ── Predict ───────────────────────────────────────────────────────
            for track in manager.track_list:
                KF.predict(track)
                track.set_t((cnt_frame - 1) * 0.1)

            # ── Associate and update ──────────────────────────────────────────
            association.associate_and_update(manager, meas_list_lidar, KF)
            association.associate_and_update(manager, meas_list_cam, KF)

            # ── Save results ──────────────────────────────────────────────────
            result_dict = {track.id: track for track in manager.track_list}
            manager.result_list.append(copy.deepcopy(result_dict))
            all_labels.append([frame.laser_labels, valid_label_flags])

            cnt_frame += 1

        except StopIteration:
            break

    print(f'\n  Processed {cnt_frame - frame_start} frames. Plotting RMSE...')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_rmse(manager, all_labels, configs_det, save_path=save_path)
    print(f'  Saved → {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Which step to run (1=EKF, 2=TrackMgmt, 3=Assoc, 4=Fusion)')
    args = parser.parse_args()
    run_step(args.step)
