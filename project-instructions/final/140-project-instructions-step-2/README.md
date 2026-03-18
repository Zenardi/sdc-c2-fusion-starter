# Project Instructions Step 2

> Part of: **Final Project: Sensor Fusion and Object Tracking**

## Additional Content

### Project Instructions Step 2

Make sure to refer to the [project rubric](https://learn.udacity.com/rubric/3006) to ensure all tasks are completed.
#### What is this task about?

In Step 2 of the final project, you will implement the track management to initialize and delete tracks, set a track state and a track score.

#### Task preparation
In addition to the settings from Step 1, apply the following settings in `loop_over_dataset.py`:
- Set `show_only_frames = [65, 100]` in order to limit the sequence to frames 65 to 100. This is the time span where a single object appears and then disappears, so we can use it for track initialization and deletion.
- Set `configs_det.lim_y = [-5, 15]` to limit the y-range and remove other targets left and right of our target. 

#### Where to find this task?
This task involves writing code within the file `student/trackmanagement.py`. Please ignore TODOs for other steps in other files for now.

#### Your task
- In the `Track` class, replace the fixed track initialization values by initialization of `track.x` and `track.P` based on the input `meas`, which is an unassigned lidar measurement object of type `Measurement`. Transform the unassigned measurement from sensor to vehicle coordinates with the `sens_to_veh` transformation matrix implemented in the `Sensor` class. Initialize the track state with 'initialized' and the score with `1./params.window`, where `window` is the window size parameter, as learned in the track management lesson.
- In the `Trackmanagement` class, implement the `manage_tracks()` function to complete the following tasks:
 - Decrease the track score for unassigned tracks.
 - Delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful). Note that you can delete tracks by calling the given function `delete_track()`, which will remove a track from `track_list`.
- In the `Trackmanagement` class, implement the `handle_updated_track()` function to complete the following tasks:
 - Increase the track score for the input `track`.
 - Set the track state to 'tentative' or 'confirmed' depending on the track score.
- Use `numpy.matrix()` for all matrices as learned in the exercises.

#### What should the result be?
If you have implemented everything correctly, the visualization shows that a new track is initialized automatically where unassigned measurements occur, the true track is confirmed quickly, and the track is deleted after it has vanished from the visible range. You can see that the track has been deleted if the console output says 'deleting track no. 0'. There is one single track without track losses in between, so the RMSE plot should show a single line.
Make sure to successfully complete this step and save the RMSE plot before moving to the next. 

#### Hints
- The parameter `params.delete_threshold = 0.6` in is only meant for deleting confirmed tracks, as stated in the comment. If you use it to delete initialized or tentative tracks, they might be deleted right after initialization because the track score is still too low. Therefore, delete initialized or tentative tracks only if `track.P[0,0]` or `track.P[1,1]` is bigger than `params.max_P`, or if the score gets very low.
- After the object has disappeared from the visible range, it might take some time until the track is deleted. This is okay because in theory the object is still there, so the track management tries to predict the track further on. Just make sure that the track is deleted eventually.
- In this project, the lidar measurements are given in vehicle coordinates, so you could skip the coordinate transformation with the `sens_to_veh` transformation matrix, for lidar it is just the identity matrix. However, I would recommend to include the transformation nevertheless in order to keep your fusion system generic and have the possibility to include other sensors later. The results will be the same with or without transformation though.
- Note that the RMSE is quite high in this scenario (around 0.8), also the green boxes don't fit the car in the image very well. This is because the lidar detections contain a y-offset. If the input has a systematic offset, the Kalman filter cannot compensate it because we assume zero-mean data. It is part of the real-world challenges that our assumptions about the data are not always met. We can, however, compensate this offset through sensor fusion once we include other sensors. But for now, don't worry about it.
- From `misc/params.py`, you can use the following parameters: `dim_state`, `sigma_p44`, `sigma_p55`, `sigma_p66`, `window`, `confirmed_threshold`, `delete_threshold`, `max_P`.
