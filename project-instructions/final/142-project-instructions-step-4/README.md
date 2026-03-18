# Project Instructions Step 4

> Part of: **Final Project: Sensor Fusion and Object Tracking**

## Images

![RMSE for the three valid tracks in Sequence 1. Track zero and track one start at ~t=0, with track 10 starting at ~t=10. The RMSE's bounce around a bit, with track zero's mean 0.15, track 1's mean 0.12, and track 10's mean 0.19.](images/rmse-s1-sigma-01-thresh-05.png)
*RMSE for the three valid tracks in Sequence 1*

## Additional Content

### Project Instructions Step 4

Make sure to refer to the [project rubric](https://learn.udacity.com/rubric/3006) to ensure all tasks are completed.
#### What is this task about?

In Step 4 of the final project, you will implement the nonlinear camera measurement model. You will finally complete the sensor fusion module for camera-lidar fusion!

#### Task preparation
The settings are the same as for Step 3.

#### Where to find this task?
This task involves writing code within the file `student/measurements.py`. 

#### Your task
- In the `Sensor` class, implement the function `in_fov()` that checks if the input state vector `x` of an object can be seen by this sensor. The function should return `True` if `x` lies in the sensor's field of view, otherwise `False`. Don't forget to transform from vehicle to sensor coordinates first. The sensor's field of view is given in the attribute `fov`.
- In the `Sensor` class, implement the function `get_hx()` with the nonlinear camera measurement function h as follows:
 - transform position estimate from vehicle to camera coordinates,
 - project from camera to image coordinates, 
 - make sure to not divide by zero, raise an error if needed,
 - return h(x).
- In the `Sensor` class, simply remove the restriction to lidar in the function `generate_measurement()` in order to include camera as well.
- In the `Measurement` class, initialize camera measurement objects including `z`, `R`, and the sensor object `sensor`.
- After completing these steps, make a movie to showcase your tracking results! You can simply do so by setting `exec_visualization = ['show_tracks', 'make_tracking_movie']` in `loop_over_dataset.py` and re-running the tracking loop.

#### What should the result be?
If you have implemented everything correctly, the tracking loop now updates all tracks with lidar measurements, then with camera measurements. The console output shows lidar updates followed by camera updates. The visualization shows that the tracking performs well, again no confirmed ghost tracks or track losses should occur. The RMSE plot should show at least three confirmed tracks. Two of the tracks should be tracked from beginning to end of the sequence (0s - 200s) without track loss. The mean RMSE for these two tracks should be below 0.25.

#### Hints
- Note that some attributes are only used from lidar measurements, for example `width` or `yaw` don't have to be set for camera measurements here (even though you could measure them with a camera).
- If more tracks are deleted now than before, maybe the check whether an object is in the camera's field of view contains an error. The score should only be decreased when an object is inside the FOV but not detected. On the other hand, the score should remain the same if an object is outside the FOV.
- Note that for simplicity, we only use one front lidar and one front camera from the Waymo Open Dataset. In reality, there are many additional sensors available that we could use.
- From `misc/params.py`, you should load the following parameters: `sigma_cam_i`, `sigma_cam_j`.
