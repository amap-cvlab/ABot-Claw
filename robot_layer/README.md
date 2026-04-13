# Robot Client

`robot_client` contains the runtime code on the robot side.

This part of the project is developed on top of ROS and is responsible for connecting high-level agent outputs to real robot execution, including motion control, status feedback, and device-side integration.

## Scope

- Robot-side runtime logic
- ROS-based communication and integration
- Execution bridge between AI services and physical robots

## Robot Categories

The current robot client code is organized into three embodiment categories:

1. **Robot Arm**
  - Manipulation-oriented control logic
  - Typical tasks: grasping, placing, and arm motion execution
2. **Quadruped**
  - Locomotion and mobility control for four-legged robots
  - Typical tasks: walking, patrol, and movement coordination
3. **Humanoid**
  - Full-body or upper-body task execution for humanoid platforms
  - Typical tasks: multi-step actions, navigation + manipulation collaboration

## Notes

- This directory focuses on robot execution and ROS integration. Please install the corresponding ROS driver according to your robot type.
- Service-side perception/reasoning components are maintained under `Services/`.
- Hardware-specific launch/config details should be documented in each submodule.

