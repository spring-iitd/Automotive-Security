# Traffic Generation and Replay in CARLA

This provides a setup to generate and replay autonomous driving traffic using CARLA. It uses behavior agents to simulate realistic driving behavior and ensures reproducible traffic patterns through timestamp-based replay.

## Overview

The workflow is automated using the `generate_replay_agent.sh` script, which performs the following steps:

1. **Traffic Generation**  
   Runs `log_gen_agent.py`, which uses CARLA's Behavior Agent to simulate two vehicles in normal mode and log their vehicle control object and timestamp data.

2. **Traffic Replay**  
   Runs `log_replay_agent.py`, which replays the previously generated traffic using the exact same timestamps.

## Simulation Details

- **Agents Used**: Behavior Agent in normal mode for both vehicles
- **Duration**: 65 seconds
- **Tick Interval**: 4 milliseconds per tick

## Limitation

- **Euclidean distance between the two vehicles is not identically replicated**.
## Usage

To run the full simulation and replay:

```bash
generate_replay_agent.sh
