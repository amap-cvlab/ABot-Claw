# System Logger

Unified state recording and rewind orchestration.

## Overview

The system logger provides:
- **Unified state recording** from all robot subsystems (base, arm, gripper)
- **Coordinated rewind** that sends commands to each server in sync
- **Waypoint storage** with efficient threshold-based filtering
- **Workspace boundary monitoring** for safety

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SystemLogger                            │
│  Records unified state from all backends at configurable Hz  │
│  Stores UnifiedWaypoint objects with base, arm, gripper data │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   RewindOrchestrator                         │
│  Reads waypoints from SystemLogger                          │
│  Sends coordinated commands to backends during rewind       │
└─────────┬───────────────┬─────────────────┬─────────────────┘
          │               │                 │
          ▼               ▼                 ▼
    ┌───────────┐   ┌───────────┐    ┌────────────┐
    │ base      │   │ franka    │    │ gripper    │
    │ _server   │   │ _server   │    │ _server    │
    └───────────┘   └───────────┘    └────────────┘
```

## Package Structure

```
system_logger/
├── system_logger/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration dataclasses
│   ├── waypoint.py           # UnifiedWaypoint dataclass
│   ├── logger.py             # SystemLogger (state recording)
│   └── rewind_orchestrator.py # RewindOrchestrator (coordinated rewind)
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

```bash
cd system_logger
pip install -e .
```

## Usage

### Recording State

```python
from system_logger import SystemLogger, LoggerConfig

# Create logger with custom config
config = LoggerConfig(
    record_interval=0.1,           # 10 Hz
    base_position_threshold=0.05,  # 5 cm
    arm_threshold=0.05,            # ~3 degrees
)
logger = SystemLogger(config)

# Start recording (provide state function)
async def get_state():
    return {
        "base": {"pose": [x, y, theta]},
        "arm": {"q": [q0, q1, q2, q3, q4, q5, q6], "ee_pose": [...]},
        "gripper": {"position": 128, "position_mm": 40.0},
    }

await logger.start(get_state)

# ... robot operates ...

# Get recorded waypoints
waypoints = logger.get_waypoints()
print(f"Recorded {len(waypoints)} waypoints")

# Stop recording
await logger.stop()
```

### Coordinated Rewind

```python
from system_logger import RewindOrchestrator, RewindConfig

# Create orchestrator
config = RewindConfig(
    settle_time=0.5,       # Wait 0.5s at each waypoint
    command_rate=10.0,     # Send commands at 10 Hz
    rewind_base=True,
    rewind_arm=True,
    rewind_gripper=False,
)
orchestrator = RewindOrchestrator(logger, config)

# Set backends
orchestrator.set_backends(
    base_backend=base_backend,
    arm_backend=arm_backend,
    gripper_backend=gripper_backend,
)

# Rewind 10% of trajectory
result = await orchestrator.rewind_percentage(10.0)
print(f"Rewound {result.steps_rewound} steps")

# Rewind to specific waypoint
result = await orchestrator.rewind_to_waypoint(idx=50)

# Rewind to last safe position (inside workspace)
result = await orchestrator.rewind_to_safe()

# Full reset to home
result = await orchestrator.reset_to_home()
```

### Selective Component Rewind

```python
# Rewind base only
result = await orchestrator.rewind_percentage(10.0, components=["base"])

# Rewind arm only
result = await orchestrator.rewind_percentage(10.0, components=["arm"])

# Rewind both
result = await orchestrator.rewind_percentage(10.0, components=["base", "arm"])
```

### Dry Run (Preview)

```python
# See what would happen without executing
result = await orchestrator.rewind_percentage(10.0, dry_run=True)
print(f"Would rewind {result.steps_rewound} steps")
for wp in result.waypoints_executed:
    print(f"  Waypoint {wp['index']}: base={wp['base_pose']}")
```

## UnifiedWaypoint

The `UnifiedWaypoint` dataclass stores complete robot state:

```python
@dataclass
class UnifiedWaypoint:
    t: float                    # Timestamp
    index: int                  # Waypoint index

    # Base state
    base_pose: List[float]      # [x, y, theta] meters, radians
    base_velocity: List[float]  # [vx, vy, wz] m/s, rad/s

    # Arm state
    arm_q: List[float]          # Joint positions [q0..q6] radians
    arm_dq: List[float]         # Joint velocities rad/s
    ee_pose: List[float]        # 4x4 matrix (16 floats, column-major)
    ee_pose_world: List[float]  # EE pose in world frame
    ee_wrench: List[float]      # [fx, fy, fz, tx, ty, tz]

    # Gripper state
    gripper_position: int       # 0-255
    gripper_width: float        # meters
    gripper_object_detected: bool

    # Metadata
    tags: List[str]             # Optional tags
    metadata: Dict[str, Any]    # Additional data
```

## Configuration

### LoggerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_waypoints` | 10000 | Max waypoints in buffer (FIFO) |
| `record_interval` | 0.1 | Recording interval in seconds |
| `base_position_threshold` | 0.05 | Min base movement to record (m) |
| `base_orientation_threshold` | 0.1 | Min rotation to record (rad) |
| `arm_threshold` | 0.05 | Min arm movement to record (rad) |

### RewindConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `settle_time` | 0.5 | Time at each waypoint (s) |
| `command_rate` | 10.0 | Command rate during rewind (Hz) |
| `rewind_base` | True | Whether to rewind base |
| `rewind_arm` | True | Whether to rewind arm |
| `rewind_gripper` | False | Whether to rewind gripper |
| `safety_margin` | 0.1 | Safety margin for boundary (m) |

### WorkspaceBounds

```python
from system_logger.config import WorkspaceBounds

bounds = WorkspaceBounds(
    base_x_min=-2.0,
    base_x_max=2.0,
    base_y_min=-2.0,
    base_y_max=2.0,
)

# Check if position is safe
is_safe = bounds.is_base_in_bounds(x, y, margin=0.1)

# Get distance to boundaries
distances = bounds.base_distance_to_boundary(x, y)
```

## Integration with Agent Server

The system logger is designed to integrate with the agent_server:

```python
# In server.py
from system_logger import SystemLogger, RewindOrchestrator

# Create logger and orchestrator
system_logger = SystemLogger()
rewind_orchestrator = RewindOrchestrator(system_logger)

# Set backends
rewind_orchestrator.set_backends(
    base_backend=base_backend,
    arm_backend=franka_backend,
    gripper_backend=gripper_backend,
)

# Start recording during server startup
await system_logger.start(state_aggregator.state)

# Rewind routes use the orchestrator
@router.post("/rewind/percentage")
async def rewind_percentage(req: RewindRequest):
    result = await rewind_orchestrator.rewind_percentage(req.percentage)
    return result
```
