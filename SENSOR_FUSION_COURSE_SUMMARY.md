# Sensor Fusion Course Material - Comprehensive Summary

## Course Overview
This is a complete course on **Sensor Fusion and Multi-Target Tracking for Autonomous Vehicles** organized in three major parts:
1. **Kalman Filter Basics** (Lessons 60-85): Foundational concepts of Gaussian distributions and filtering
2. **Extended Kalman Filter Sensor Fusion** (Lessons 87-110): Fusing LiDAR and camera measurements
3. **Multi-Target Tracking** (Lessons 111-135): Track management, data association, and fusion pipeline

---

# PART 1: KALMAN FILTER FUNDAMENTALS (Lessons 60-85)

## Core Mathematical Foundation

### Gaussian Distributions (Lesson 62)
- **Definition**: Continuous probability distribution with two key parameters:
  - **Mean (μ)**: Center of the distribution
  - **Variance (σ²)**: Width/spread of the distribution
- **Formula**: P(x) = (1/√(2πσ²)) × exp(-1/2 × ((x - μ)² / σ²))
- **Key Property - Unimodality**: Single peak, symmetrical, exponential drop-off on both sides
- **Use in Kalman filters**: Represents uncertainty about system state

### Motion Update (Prediction) (Lesson 75)
- **Concept**: When an object moves with uncertainty, both position and uncertainty change
- **Mean Update**: μ' = μ + u (where u is motion command)
- **Variance Update**: σ'² = σ² + r² (where r² is motion uncertainty)
- **Intuition**: Moving increases uncertainty because actual acceleration is unknown
- **Example**: If current position has μ=8, σ²=4, and object moves 10m with uncertainty 6:
  - New mean: 8 + 10 = 18
  - New variance: 4 + 6 = 10

### Measurement Update (Bayes Rule)
- **Concept**: Combining predicted state with sensor measurement
- **Method**: Gaussian multiplication (equivalent to Bayes rule)
- **Result**: Creates a new Gaussian that's sharper than either input (reduces uncertainty)
- **Principle**: More precise input (smaller variance) has more influence on the result

### Kalman Filter Code Implementation
- **Basic Structure**:
  1. Initialize state estimate and uncertainty
  2. LOOP:
     - **Predict**: Apply motion model → increases uncertainty
     - **Update**: Apply measurement → decreases uncertainty
     - Repeat with new measurement

---

# PART 2: EXTENDED KALMAN FILTER SENSOR FUSION (Lessons 87-110)

## System Overview
The Extended Kalman Filter combines measurements from multiple sensors (LiDAR + Camera) to track objects over time.

### System State Representation
- **4D State Vector**: x = [px, py, vx, vy]ᵀ
  - px, py: 2D position in vehicle coordinates
  - vx, vy: 2D velocity (constant velocity assumption)
- **Advantages**: Simple yet effective for autonomous vehicle tracking
- **Limitation**: Assumes constant velocity; acceleration is handled via process noise

---

## PREDICTION STEP (State Transition)

### Linear Motion Model
- **State Transition Matrix F(Δt)**:
```
F = [1  0  Δt  0 ]
    [0  1  0  Δt ]
    [0  0  1   0 ]
    [0  0  0   1 ]
```
- **Prediction**: x⁻ = F × x⁺
- **Interpretation**: 
  - p' = p + v × Δt (position advances by velocity × time)
  - v' = v (velocity stays constant)

### Process Noise Covariance Q
- **Purpose**: Accounts for unmodeled acceleration and system uncertainties
- **Design Parameter**: q (in m²/s⁴, represents expected maximum acceleration)
- **Discretized Formula**:
```
Q = [Δt⁴/4×q   0      Δt³/2×q   0    ]
    [0         Δt⁴/4×q  0      Δt³/2×q]
    [Δt³/2×q   0       Δt²×q     0    ]
    [0         Δt³/2×q  0      Δt²×q  ]
```
- **Selection Guidelines**:
  - Emergency braking: q ≈ (8 m/s²)²
  - Normal highway: q ≈ (3 m/s²)²
  - Higher q = expect more aggressive acceleration

### Covariance Prediction
- **Formula**: P⁻ = F × P⁺ × Fᵀ + Q
- **Meaning**: Transform current covariance, add process noise
- **Effect**: Uncertainty always increases between measurements

---

## MEASUREMENT MODELS

### LiDAR Measurement Model (Linear)

#### Measurement Matrix H
```
H = [1  0  0  0]
    [0  1  0  0]
```
- **Input**: 4D state [px, py, vx, vy]
- **Output**: 2D measurement z = [px, py] (position only)
- **Why**: LiDAR directly measures position, not velocity

#### Measurement Covariance R
- **Definition**: Uncertainty in LiDAR measurements
- **Typical Values**: Diagonal matrix with variances for x, y position
- **Source**: LiDAR sensor specifications (e.g., 0.1-0.2m standard deviation)

#### LiDAR Update (Linear Kalman Filter)
- **Residual**: γ = z - H × x⁻ (predicted vs measured position)
- **Residual Covariance**: S = H × P⁻ × Hᵀ + R
- **Kalman Gain**: K = P⁻ × Hᵀ × S⁻¹
  - Determines how much to trust measurement vs prediction
- **State Update**: x⁺ = x⁻ + K × γ
- **Covariance Update**: P⁺ = (I - K × H) × P⁻

---

### Camera Measurement Model (Nonlinear)

#### Nonlinear Measurement Function h(x)
Camera projects 3D vehicle coordinates to 2D image coordinates:
```
h(x) = [ci - (fi × py / px)]
       [cj - (fj × pz / px)]
```
Where:
- ci, cj: Principal point (image center from camera calibration)
- fi, fj: Focal lengths (from intrinsic calibration)
- px, py, pz: 3D position in camera frame
- Result: 2D pixel coordinates [i, j] in image

#### Why Nonlinear?
- **Problem**: Gaussian input doesn't remain Gaussian after nonlinear transformation
- **Solution**: Use Extended Kalman Filter with linearization

---

## EXTENDED KALMAN FILTER (EKF)

### Linearization via Taylor Expansion
- **Concept**: Approximate nonlinear function with a line (tangent)
- **1st Order Taylor Expansion**:
```
h(x) ≈ h(μ) + h'(μ) × (x - μ)
```
Where h'(μ) is the first derivative (slope) at mean μ

- **Intuition**: Works well when state is close to mean μ
- **Result**: Linearized function preserves Gaussian properties

### Jacobian Matrix HJ
- **Definition**: Matrix of first-order partial derivatives of measurement function
- **For Camera (2×6 matrix)**:
```
HJ = [∂h1/∂px  ∂h1/∂py  ∂h1/∂pz  ∂h1/∂vx  ∂h1/∂vy  ∂h1/∂vz]
     [∂h2/∂px  ∂h2/∂py  ∂h2/∂pz  ∂h2/∂vx  ∂h2/∂vy  ∂h2/∂vz]
```
- **Computed Values for Camera**:
```
HJ = [fy·py/px²    -fi/px      0          0  0  0]
     [fj·pz/px²     0         -fj/px      0  0  0]
```
- **Velocity Derivatives**: All zero because camera doesn't measure velocity

### EKF Update Steps

#### Prediction (Same as Linear Kalman)
- Use linear motion model F
- x⁻ = F × x⁺
- P⁻ = F × P⁺ × Fᵀ + Q

#### Measurement Update (Modified for Nonlinearity)
1. **Calculate h(x)**: Transform state to measurement space (e.g., 3D→2D image)
2. **Residual**: γ = z - h(x⁻) [Note: uses h function, not H matrix]
3. **Jacobian**: Calculate HJ at current state x⁻
4. **Residual Covariance**: S = HJ × P⁻ × HJᵀ + R
5. **Kalman Gain**: K = P⁻ × HJᵀ × S⁻¹
6. **State Update**: x⁺ = x⁻ + K × γ
7. **Covariance Update**: P⁺ = (I - K × HJ) × P⁻

---

## COORDINATE TRANSFORMATIONS

### Vehicle vs Sensor Coordinates
- **Vehicle Frame**: Origin at rear axle, X forward, Y left, Z up (right-handed)
- **Sensor Frames**: Each sensor has its own frame (may be rotated/translated)
- **Problem**: LiDAR and camera measurements arrive in sensor coordinates
- **Solution**: Transform to vehicle frame before fusion

### Transformation Matrix T
Combines rotation and translation in homogeneous coordinates:
```
T = [R11  R12  R13  t1]
    [R21  R22  R23  t2]
    [R31  R32  R33  t3]
    [0    0    0    1 ]
```
Where:
- Upper-left 3×3: Rotation matrix M_rot
- Right column: Translation vector t

### Transformation Process
1. Convert measurement to homogeneous coordinates: [z1, z2, z3, 1]ᵀ
2. Apply transformation: p_veh = T × z_homogeneous
3. Drop last component to get [px, py, pz]ᵀ in vehicle frame

### Example: 45° Rotated LiDAR, 4m Forward
```
T = [cos(π/4)  -sin(π/4)  0   4]      [0.7   -0.7   0   4]
    [sin(π/4)   cos(π/4)  0   0]  =   [0.7    0.7   0   0]
    [0          0         1   0]      [0      0     1   0]
    [0          0         0   1]      [0      0     0   1]
```

---

# PART 3: MULTI-TARGET TRACKING (Lessons 111-135)

## Track Management System

### Overview
Handle multiple objects, missed detections, false positives, and track lifecycle.

### Track Initialization

#### From LiDAR Measurement
1. **Input**: 3D measurement z in sensor coordinates
2. **Transform**: Convert to vehicle coordinates using sensor calibration T
3. **Initialize State**: x = [px, py, pz, 0, 0, 0]ᵀ
   - Position from measurement
   - Velocity = 0 (unknown)
4. **Initialize Covariance**:
   - **Position Part**: P_pos = M_rot × R × M_rotᵀ
     - Rotate measurement covariance R to vehicle frame
   - **Velocity Part**: P_vel = diagonal matrix with large values
     - High uncertainty since velocity unmeasured
   - **Combined**: P0 = block diagonal of [P_pos, P_vel]
5. **Assign ID**: Increment track counter for unique identification

#### Covariance Initialization Detail
- **Measurement Covariance**: R represents LiDAR position uncertainty
- **Rotation**: Rotate ellipse of uncertainty from sensor→vehicle frame
- **Velocity Uncertainty**: Large diagonal values (e.g., 10² m²/s²) until confirmed by multiple measurements

---

### Track Score and State

#### Track Score
- **Definition**: Confidence measure from 0 (false positive) to 1 (real object)
- **Simple Formula**: score = (# detections in last n frames) / n
- **Example with n=5 frames**:
  - Frame 1: 1 detection → score = 0.2
  - Frame 2: 2 detections → score = 0.4
  - Frame 3: 3 detections → score = 0.6
  - Frame 4: 4 detections → score = 0.8
  - Frame 5: 5 detections → score = 1.0
- **False Positives**: Ghost tracks drop score when not re-detected

#### Track State (Based on Score)
- **Initialized**: score < 0.3 (uncertain, new detection)
- **Tentative**: 0.3 ≤ score < 0.8 (probably real but not confirmed)
- **Confirmed**: score ≥ 0.8 (high confidence, real object)
- **Deleted**: score drops to 0 or covariance becomes too large

---

### Visibility Reasoning

#### Problem in Sensor Fusion
- LiDAR and camera have different fields of view (FOV)
- If object visible only to LiDAR but not camera:
  - Without visibility: score oscillates (LiDAR ↑, camera ↓)
  - Result: Track never becomes "confirmed"

#### Solution
- **Check Visibility**: Determine if object is in each sensor's FOV
- **Update Score Only If Visible**: Only decrease score if object is expected but not detected
- **Benefits**: Each sensor contributes independently to confidence

#### Visibility Check
Simple geometric test if object position is in sensor FOV:
- Distance and angle constraints specific to each sensor
- More advanced: detection probability per sensor

---

## DATA ASSOCIATION (Assigning Measurements to Tracks)

### The Problem
- Multiple tracks, multiple measurements
- Need to determine which measurement came from which track
- Assumption: Each track generates ≤1 measurement, each measurement from ≤1 track

### Mahalanobis Distance
- **Purpose**: Measure how well a measurement matches a track prediction
- **Formula**:
```
d(x, z) = γᵀ × S⁻¹ × γ
        = (z - h(x))ᵀ × S⁻¹ × (z - h(x))
```
Where:
- γ = residual (predicted vs measured)
- S = residual covariance = H×P×Hᵀ + R
- Accounts for uncertainty of both track and measurement

- **Advantage over Euclidean**: 
  - Euclidean: |z - x| treats position directly
  - Mahalanobis: Weights by covariance (big uncertainty → smaller distance)
  - Example: Uncertain track with large ellipse is better match than precise track with small ellipse

### Simple Nearest Neighbor (SNN) Association
1. **Calculate all distances**: D[i,j] = Mahalanobis distance (track i → measurement j)
2. **Find minimum**: d_min = smallest distance in D
3. **Associate**: Update track with that measurement
4. **Remove**: Delete row and column from D
5. **Repeat**: Until no more valid associations

#### Limitations
- Hard decisions in ambiguous situations
- Not globally optimal
- Can lead to cascading errors

#### Alternatives (Not Covered Here)
- **Global Nearest Neighbor (GNN)**: Globally optimal assignment
- **Probabilistic Data Association (PDA)**: Soft probabilities, avoid hard decisions

---

### Gating (Reduce Computational Complexity)

#### Concept
- Create a "gate" (ellipse) around each track
- Only consider measurements inside the gate
- Set distances outside gate to ∞ (reject them)

#### Mathematical Gating
- **Test**: Is Mahalanobis distance ≤ threshold?
- **Threshold**: Inverse cumulative χ² CDF
```
d(x, z) ≤ F_χ²⁻¹(0.995 | dim_z)
```
- Parameter 0.995 = 99.5% of true measurements inside gate
- dim_z = measurement dimension (2 for LiDAR, 2 for camera)

#### Example Matrix with Gating
After calculating all distances, apply gating:
```
Original:         After Gating:
[0.5  1.2  3.1]  [0.5   ∞    ∞  ]
[0.8  0.4  2.1]  [0.8  0.4  2.1]
[4.2  3.8  0.6]  [∞    ∞    0.6]
```
Measurements outside gates marked ∞ are ignored.

---

## COMPLETE FUSION FLOW CHART

### Step-by-Step Process for Each Measurement Cycle

1. **PREDICT ALL TRACKS**
   - For each track: x⁻ = F × x⁺, P⁻ = F×P⁺×Fᵀ + Q
   - Advance to new measurement timestamp

2. **ASSOCIATE MEASUREMENTS TO TRACKS**
   - Calculate Mahalanobis distances
   - Apply gating to reduce complexity
   - Use SNN to find associations
   - Result: Track-measurement pairs

3. **UPDATE ASSOCIATED TRACKS**
   - For LiDAR: x⁺ = x⁻ + K×(z - H×x⁻)
   - For Camera: x⁺ = x⁻ + K×(z - h(x⁻))
   - Increase track score if visible
   - Update track state (initialized→tentative→confirmed)

4. **HANDLE UNASSOCIATED TRACKS**
   - If in FOV but not detected: Decrease track score
   - If score → 0: Delete track

5. **INITIALIZE NEW TRACKS**
   - For unassociated measurements: Create new tracks
   - Initialize with position, zero velocity, high uncertainty

6. **DELETE LOW-CONFIDENCE TRACKS**
   - If score < threshold (e.g., 0.1): Remove
   - If covariance too large: Remove

7. **REPEAT**: Go to step 1 with new measurement

---

# KEY FORMULAS SUMMARY

## Kalman Filter (Linear)
```
Prediction:
  x⁻ = F × x⁺
  P⁻ = F × P⁺ × Fᵀ + Q

Update:
  γ = z - H × x⁻
  S = H × P⁻ × Hᵀ + R
  K = P⁻ × Hᵀ × S⁻¹
  x⁺ = x⁻ + K × γ
  P⁺ = (I - K × H) × P⁻
```

## Extended Kalman Filter (Nonlinear)
```
Prediction: (Same as Linear)

Update:
  γ = z - h(x⁻)         [Uses function h, not matrix H]
  HJ = Jacobian of h at x⁻
  S = HJ × P⁻ × HJᵀ + R
  K = P⁻ × HJᵀ × S⁻¹
  x⁺ = x⁻ + K × γ
  P⁺ = (I - K × HJ) × P⁻
```

## Mahalanobis Distance
```
d(x, z) = (z - h(x))ᵀ × S⁻¹ × (z - h(x))
```

## Gating Condition
```
d(x, z) ≤ F_χ²⁻¹(0.995 | dim_z)
```

---

# IMPLEMENTATION GUIDELINES

## Project Architecture
1. **Sensor Input**: LiDAR point clouds + Camera images
2. **Detection Module**: Extract object bounding boxes
3. **Sensor Fusion Module**:
   - Track manager (initialize, delete, score tracks)
   - Prediction (apply motion model)
   - Association (match measurements to tracks)
   - Update (fuse measurements)
4. **Output**: Tracks with position, velocity, confidence

## Common Tuning Parameters
- **Process Noise q**: Design parameter for expected acceleration
  - Normal: 3 m/s² → q = 9 m²/s⁴
  - Emergency: 8 m/s² → q = 64 m²/s⁴

- **Measurement Covariance R**: Sensor specifications
  - LiDAR: typically 0.1-0.3m standard deviation
  - Camera: Calibration-dependent

- **Gating Threshold**: 0.995 (99.5% of true measurements)
  - Can adjust for trade-off between false positives and computational cost

- **Track Score Thresholds**:
  - Tentative → Confirmed: 0.8
  - Delete if score < 0.1-0.2

- **Initial Velocity Uncertainty**: Large diagonal matrix
  - Typical: 10-100 m²/s²

---

# DESIGN CONSIDERATIONS

## Challenges
1. **Varying Sensor FOVs**: Different coverage → visibility reasoning needed
2. **Coordinate Transforms**: Errors in calibration → propagate through filters
3. **Real-Time Constraints**: Must process streams of measurements efficiently
4. **Handling Occlusions**: Tracks may disappear and reappear
5. **False Positives**: Clutter measurements create ghost tracks
6. **Parameter Sensitivity**: Different scenarios (highway vs urban) need different tuning

## Solutions
1. **Visibility Module**: Check each sensor's FOV before updating scores
2. **Careful Calibration**: Accurate sensor extrinsics essential
3. **Gating**: Reject unlikely associations, reduce complexity
4. **Adaptive Parameters**: Switch between parameterizations based on driving scenario
5. **Sensor Fusion**: Multiple sensors reduce false positives through redundancy
6. **Track Management**: Score and state logic with thresholds

---

# PRACTICE EXERCISES (Embedded in Course)

- Compute pitch angle resolution for LiDAR range images
- Visualize intensity channel from LiDAR
- Transform point clouds to bird's-eye view
- Implement Kalman filter equations
- Compute Jacobians for camera measurement function
- Initialize tracks and covariance matrices
- Implement track scoring and visibility checks
- Compute Mahalanobis distances
- Build association matrices with gating
- Implement complete tracking pipeline

