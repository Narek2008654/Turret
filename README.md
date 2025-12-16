# Turret

# Problem Definition — Turrent

## Task Name

**Turrent**

---

## Objective

The first task is to develop a system that, using only a **single monocular RGB camera**, learns to **assess collision risk from visual input and select an appropriate discrete action** in real time.

The system must operate **without access to true distance, depth, velocity, or trajectory information**, and must rely solely on **temporal visual cues** extracted from image sequences.

---

## Problem Statement

Given a sequence of RGB images captured over time, the system must learn to:

1. Detect and interpret approaching humans in the scene
2. Select one action from a predefined discrete action set at each time step

The system must function in environments where multiple people may be present and approaching at different speeds and directions.

---

## Input

At each time step ( t ), the system receives:

[
s_t = {I_{t-N+1}, ... , I_t}
]

Where:

* ( I_t ) is an RGB image
* ( N ) is the number of stacked consecutive frames

This stacked image state encodes:

* Motion
* Relative position
* Apparent scale changes of people
* Temporal context for speed inference

---

## Output

At each time step, the system outputs a single discrete action:

| Action ID | Action     |
| --------- | ---------- |
| 0         | Left  |
| 1         | Right |
| 2         | Up    |
| 3         | Down  |
| 4         | Up and Right   |
| 5         | Up and Left    |
| 6         | Down and Right |
| 7         | Down and Left  |
| 8         | Shoot          |
| 9         | Do nothing     |

Actions are executed at fixed time intervals and are mutually exclusive.

---

## Constraints

* Only monocular RGB input is available
* No depth sensors or stereo vision
* No ground-truth distance or speed labels
* No explicit time-to-collision labels in the dataset
* No manual rules or safety overrides during inference

---

## Learning Formulation

* **State** ( s_t ): stacked RGB frames
* **Action** ( a_t ): one of the discrete actions
* **Reward** ( r_t ): scalar signal encouraging increased safety and penalizing imminent collisions

The goal is to learn a policy that maximizes expected cumulative reward.

---

## Success Criteria

The task is considered successfully solved if the system:

* Reacts earlier to faster-approaching individuals
* Reduces collision events over time
* Produces stable, non-random action sequences
* Generalizes to new episodes and unseen motion patterns

---

Training Inputs (Offline Dataset)
1.1 Raw Visual Input (Primary Input)

File: frames.npy

Type: numpy.ndarray

Shape: ( 𝑇, 128, 128, 3)

Data type: uint8

Range: [0, 255]

Meaning:

T = number of time-ordered frames in one episode

Each frame is a monocular RGB image captured at a fixed frame rate

Constraint:

Frames must be in correct temporal order

Camera is fixed to the agent (egocentric view)

This is the only input used by the neural network.

1.2 Action Labels (Supervision Only)

File: actions.npy

Type: numpy.ndarray

Shape: (T,)

Data type: int64

Value range: ∈{0,1,2,3,4,5}