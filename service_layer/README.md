# Services Overview

This directory contains the core service modules used by ABot-Claw.  
Use this page as an index of available services, their purpose, and their source repositories.

## Service List


| Service       | Directory                | Core Capability                                                       | Entry Point |
| ------------- | ------------------------ | --------------------------------------------------------------------- | ----------- |
| YOLO          | `Services/YOLO`          | Object detection + depth-based distance estimation                    | `main.py`   |
| GraspAnything | `Services/GraspAnything` | Grasp-oriented detection and grasp-related inference                  | `main.py`   |
| VLAC          | `Services/VLAC`          | Vision-Language-Action critic inference (progress/completion scoring) | `main.py`   |
| SpatialMemory | `Services/SpatialMemory` | Unified spatial memory (object/place/semantic/keyframe memory)        | `main.py`   |


## Service Details

### 1) YOLO

- Directory: `Services/YOLO`
- Purpose: YOLO-based object detection with D435i depth data for object distance estimation.
- Typical use cases: online perception, object localization, and detection input for grasping/navigation.
- Source: [https://github.com/killnice/yolov5-D435i.git](https://github.com/killnice/yolov5-D435i.git)

### 2) GraspAnything

- Directory: `Services/GraspAnything`
- Purpose: Provides grasp-task-related detection and inference capabilities.
- Typical use cases: candidate target selection and pre-grasp perception for robot manipulation.
- Source: [https://github.com/graspnet/anygrasp_sdk.git](https://github.com/graspnet/anygrasp_sdk.git)

### 3) VLAC

- Directory: `Services/VLAC`
- Purpose: Provides Vision-Language-Action critic capabilities for task progress evaluation, completion judgment, and trajectory quality scoring.
- Typical use cases: reinforcement learning reward estimation, task execution monitoring, and trajectory data filtering.
- Source: [https://github.com/InternRobotics/VLAC.git](https://github.com/InternRobotics/VLAC.git)

### 4) SpatialMemory

- Directory: `Services/SpatialMemory`
- Purpose: Unified memory service for object memory, place memory, semantic frame memory, and keyframe memory, with query APIs.
- Typical use cases: long-term robot memory, target recall/localization, and task-context retrieval.

