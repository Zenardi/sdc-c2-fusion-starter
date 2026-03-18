# Repository Overview

> Part of: **Final Project: Sensor Fusion and Object Tracking**

## Additional Content

### Repository Overview
The [final project](https://github.com/udacity/nd013-c2-fusion-starter) uses the following folder structure and files:

-  You already know `loop_over_dataset.py` as the main file for our lidar detection and tracking loop, which processes each measurement frame. 
- `student/filter.py` contains the EKF class including predict and update step. You will implement the EKF in Step 1 of the project.
- `student/trackmanagement.py` includes two classes for tracking:
 - A class `Track` with attributes `x` and `P` for state and covariance. Both have fixed initial values implemented, but you will change this in Step 2.
 - A class `Trackmanagement` with a `track_list` to store all tracks, as well as methods to manage tracks. This will also be implemented in Step 2.
- `student/association.py` includes a class `Association` with logic to associate tracks to measurements and call the EKF update function with these associated measurements in `associate_and_update()`. You will implement the data association in Step 3. 
- `student/measurements.py` includes two classes for measurement handling:
 - A class `Sensor` which distinguishes between names `camera` and `lidar`. It includes the sensor's calibration data, field of view and coordinate transforms as well as the EKF measurement model.
 - A class `Measurement` with the attributes `z`and `R` for the measurement vector and corresponding covariance. The lidar measurement code is given, you will implement the camera measurement for sensor fusion in Step 4.
- `misc/params.py` includes all parameters for tracking. You should load these parameters where needed, for example the timestep, initialization parameters, track management settings, gating threshold. The parameters don't have to be modified. However, you can tune them further after completing the project to improve your tracking, e.g. by using the lidar standard deviation you evaluated in the mid-term project. I would recommend to only change parameters after completion of Step 4 to avoid additional error sources.

Please take a look at these files now to get a basic overview of the tracking files in the repository. Refer to the README and to the mid-term project instructions for more details on the other files that weren't mentioned here if you are interested.
