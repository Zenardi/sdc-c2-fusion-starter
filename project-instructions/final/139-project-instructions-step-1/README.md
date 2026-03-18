# Project Instructions Step 1

> Part of: **Final Project: Sensor Fusion and Object Tracking**

## Images

![This image shows the single track from Sequence 2 in birds-eye view and projected to the camera image.](images/single-target.png)
*Single target tracking results*

## Additional Content

### Project Instructions Step 1

Make sure to refer to the [project rubric](https://learn.udacity.com/rubric/3006) to ensure all tasks are completed.
#### What is this task about?

In Step 1 of the final project, you will implement an EKF to track a single real-world target with lidar measurement input over time! 

#### Task preparation
I have prepared a simple single-target-scenario to get you started. Make sure to carefully follow these steps, because you don't have a running track management or data association yet, so you can't run other multi-target-scenarios - but we'll get to it!

First of all, you need to apply the same model settings for the Resnet neural network as in the mid-term project. Therefore, simply copy your mid-term solution code in `student/objdet_detect.py` from the mid-term project workspace into the final project workspace. In particular, you need the model configuration settings you used in the mid-term student exercises `ID_S3_EX1-3` and `ID_S3_EX1-4`.

To select the right single-target-scenario, apply the following settings in `loop_over_dataset.py`:
- Select Sequence 2 (`training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`) by uncommenting this line in `loop_over_dataset.py` and commenting the other sequences.
- Set `show_only_frames = [150, 200]` in order to limit the sequence to frames 150 to 200. This is the time span where our single object is visible.
- Set `configs_det = det.load_configs(model_name='fpn_resnet')` to use the Resnet neural network architecture. Note that Darknet is not applicable here because it does not estimate the height.
- Set `configs_det.lim_y = [-5, 10]` to limit the y-range and remove other targets left and right of our target. To do so, uncomment the respective line. 
- Set `exec_detection = []` to skip the lidar detection for faster execution and to load the lidar results from file instead. 
- Set `exec_tracking = ['perform_tracking']` to activate tracking.
- Set `exec_visualization = ['show_tracks']` for track visualization.

#### Where to find this task?
This task involves writing code within the file `student/filter.py`. Please ignore TODOs for other steps in other files for now.

#### Your task
- The single track is already initialized for you, so don't worry about track initialization right now.
- In `student/filter.py`, implement the `predict()` function for an EKF. Implement the `F()` and `Q()` functions to calculate a system matrix for constant velocity process model in 3D and the corresponding process noise covariance depending on the current timestep dt. Note that in our case, dt is fixed and you should load it from `misc/params.py`. However, in general, the timestep might vary. At the end of the prediction step, save the resulting x and P by calling the functions `set_x()` and `set_P()` that are already implemented in `student/trackmanagement.py`. 
- Implement the `update()` function as well as the `gamma()` and `S()` functions for residual and residual covariance. You should call the functions `get_hx` and `get_H` that are already implemented in `students/measurements.py` to get the measurement function evaluated at the current state, h(x), and the Jacobian H. Note that we have a linear measurement model for lidar, so h(x)=H*x for now. You should use h(x) nevertheless for the residual to have an EKF ready for the nonlinear camera measurement model you'll need in Step 4. Again, at the end of the update step, save the resulting x and P by calling the functions `set_x()` and `set_P()` that are already implemented in `student/trackmanagement.py`. 
- Use `numpy.matrix()` for all matrices as learned in the exercises.

#### What should the result be?
If you have implemented everything correctly, the RMSE plot should show a mean RMSE of 0.35 or smaller. You can see the computed mean RMSE in the legend on the right. Make sure to successfully complete this step and save the RMSE plot before moving to the next. 

#### Hints
- We now want to track 3D objects with a constant velocity model including height estimation, so F and Q will be 6D matrices, in comparison to our 2D tracking in the lesson exercise where we assumed a flat world. Therefore, you need to implement the following matrices: [TODO: include image of Latex formulas]
- Remember from the repository overview on the last page that there is a `Track` class and a `Measurement` class. These classes define your input to the `predict()` and `update()` functions. So you can get the track data by calling `track.x` and `track.P`, the measurement data by calling `meas.z` and `meas.R`. 
Also note that the measurement has an attribute `sensor` that tells us which sensor generated this measurement, so you can get the measurement matrix by calling `meas.sensor.get_H()`. 
Take a closer look at the two classes for clarification.
- Note that you don't have a running track management yet, therefore the track state is fixed at 'confirmed' and the score remains at the initial value zero.
- From `misc/params.py`, you should load the following parameters: `dt`, `q`, `dim_state`.
