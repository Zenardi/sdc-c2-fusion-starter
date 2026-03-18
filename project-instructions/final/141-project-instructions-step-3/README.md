# Project Instructions Step 3

> Part of: **Final Project: Sensor Fusion and Object Tracking**

## Images

![This image shows three tracks from Sequence 1 in birds-eye view and projected to the camera image.](images/s1.png)
*Tracking result with three tracks from Sequence 1.*

## Additional Content

### Project Instructions Step 3

Make sure to refer to the [project rubric](https://learn.udacity.com/rubric/3006) to ensure all tasks are completed.
#### What is this task about?

In Step 3 of the final project, you will implement a single nearest 
neighbor data association to associate measurements to tracks. You will 
finally move on to multi target tracking now!

#### Task preparation
In addition to the settings from Step 2, apply the following settings in
 `loop_over_dataset.py`:
- Select Sequence 1 
(`training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`)
 by uncommenting this line in `loop_over_dataset.py` and commenting out 
the other sequences. This is a more complex scenario with multiple 
targets.
- Set `show_only_frames = [0, 200]` in order to use the whole sequence 
now.
- Set `configs_det.lim_y = [-25, 25]` to use the whole y-range including 
several targets. 

#### Where to find this task?
This task involves writing code within the file 
`student/association.py`. Please ignore TODOs for other steps in other 
files for now.

#### Your task
- In the `Association` class, implement the `associate()` function to complete the following tasks:
 - Replace `association_matrix` with the actual association matrix based on Mahalanobis distances for all tracks in the input `track_list` and all measurements in the input `meas_list`. Use the `MHD()`function to implement the Mahalanobis distance between a track and a measurement. Also, use the `gating()` function to check if a measurement lies inside a track's gate. If not, the function shall return `False` and the entry in `association_matrix` shall be set to infinity. 
 - Update the list of unassigned measurements `unassigned_meas` and unassigned tracks `unassigned_tracks` to include the indices of all measurements and tracks that did not get associated.
- In the `Association` class, implement the `get_closest_track_and_meas()` function to complete the following tasks:
 - Find the minimum entry in `association_matrix`, delete corresponding row and column from the matrix.
 - Remove corresponding track and measurement from `unassigned_tracks` and `unassigned_meas`.
 - Return this association pair between track and measurement. If no more association was found, i.e. the minimum matrix entry is infinity, return `numpy.nan` for the track and measurement.

#### What should the result be?
The association works properly if you see in the visualization that multiple tracks are updated with multiple measurements. The console output shows that each measurement is used at most once and each track is updated at most once. The visualization should show that there are no confirmed “ghost tracks” that do not exist in reality. There may be initialized or tentative “ghost tracks” as long as they are deleted after several frames.
Make sure to successfully complete this step and save the RMSE plot before moving to the next. If you still saw some initialized or tentative ghost tracks here, let's see if we can deplausibilize them through sensor fusion with camera in the next step!

#### Hints
- If the console output shows that a single measurement has been used several times, there is an error in the association matrix or in the deletion of used rows and columns. Printing the association matrix for debugging might help.
- From `misc/params.py`, you should load the parameter `gating_threshold`.
