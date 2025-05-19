# CAN Data Generation and Processing for CARLA Simulation

This repository provides a pipeline to simulate benign vehicle behavior in CARLA, generate CAN traffic over a virtual CAN bus (`vcan0`), apply realistic timestamp adjustments through jitter.

---

## Simulation Overview

- Two **benign vehicles** are spawned in CARLA with a **25-meter gap**.
- Both vehicles are controlled via **BasicAgent** and limited to **100 km/h**.
- The simulation runs for **22.5 seconds**, during which CAN traffic is generated based on vehicle control objects.

---

## Workflow Summary

The main script `generate_data_car.sh` automates the following:

1. Prepares log and graph directories.
2. Starts `candump` to log CAN traffic on `vcan0`.
3. Runs the vehicle simulation (`benign_car_simulation.py`).
4. Stops logging and processes the output using:
   - `update_can_logs.py` — aligns CAN timestamps based on the timestamps of vehicle control objects.
   - `add_random_jitter.py` — adds jitter to simulate real-world timing noise.
   - `arbitration.py` — performs CAN arbitration.
   - `plot_order_of_packets.py` — generates plots of CAN ID arrival order.

---

## Timestamp Jittering

To better reflect real-world CAN behavior, `add_random_jitter.py` introduces small random jitter (e.g., ±1 ms) to each CAN message timestamp. After applying the jitter:

- All messages are **re-sorted chronologically**.
- The result (`can_logs_jittered.log`) mimics natural timestamp fluctuations.

---

## Output Files

- `Logs/can_data_logs.log`: Raw CAN log from simulation.
- `Logs/updated_can_data_logs.log`: Timestamps synced with the vehicle control objects.
- `Logs/can_logs_jittered.log`: Timestamp-jittered CAN log.
- `Graphs/*.png`: Visualizations of CAN message order and other relevant graphs.

---

## Run the Pipeline

```bash
generate_data_car.sh
