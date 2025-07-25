from collections import deque
import glob
from multiprocessing import Event, Process
import os
import queue
import random
import re
import sys
import threading
import time
import timeit

import pygame
import carla
import can
import cantools
from plot_graphs import plot_benign_timeline_near_dos, plot_diff, plot_euclid_diff, plot_euclid_diff_by_index_only, plot_euclid_diff_single_function, plot_gen_path_only_from_list, plot_gen_vs_rep_paths_from_files, plot_spoof_timeline, plot_vc_index, plot_vc_time

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + '/carla')
except IndexError:
    pass

from agents.navigation.custom_agent import CustomAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent 

class CAN_Data_Logger(object):
    def __init__(self):
        self.dbc = cantools.database.load_file('./bighonda.dbc')
        self.steer_message = self.dbc.get_message_by_name('STEERING_SENSORS')
        self.gear_message = self.dbc.get_message_by_name('GEARBOX')
        self.throttle_brake_message = self.dbc.get_message_by_name('POWERTRAIN_DATA')
        self.blinker_message = self.dbc.get_message_by_name('SCM_FEEDBACK')
        self.headlight_message = self.dbc.get_message_by_name('STALK_STATUS')
        self.beam_message = self.dbc.get_message_by_name('STALK_STATUS_2')
        self.handbrake_message = self.dbc.get_message_by_name('VSA_STATUS')
        self.speed_message = self.dbc.get_message_by_name('CAR_SPEED')
        self.pattern = r"ID:\s+([0-9a-fA-F]+).*DL:\s+\d+\s+([0-9a-fA-F\s]+)"
        self.time_diff = []
        self.bus = can.Bus(channel='vcan0', interface='socketcan')

    def log_steer_data(self, steer_data):
        encoded_steer_data = self.steer_message.encode({'STEER_ANGLE': steer_data})
        message = can.Message(arbitration_id=self.steer_message.frame_id, data=encoded_steer_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    def log_throttle_brake_data(self, throttle_data, brake_data):
        encoded_throttle_brake_data = self.throttle_brake_message.encode({'PEDAL_GAS': throttle_data, 'ENGINE_RPM': 0, 'GAS_PRESSED': 0, 
                                                   'ACC_STATUS': 0, 'BOH_17C': 0, 'BRAKE_SWITCH': 0, 
                                                   'BOH2_17C': 0, 'BRAKE_PRESSED': brake_data})
        message = can.Message(arbitration_id=self.throttle_brake_message.frame_id, data=encoded_throttle_brake_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)
       
    def log_gear_data(self, gear_data, manual_gear_shift):
        manual_gear_shift = 1 if (manual_gear_shift == True) else 0   
        if manual_gear_shift:    #1 for manual and 0 for auto transmission
            encoded_gear_data = self.gear_message.encode({'GEAR_SHIFTER': 1, 'GEAR': gear_data})
        else:
            encoded_gear_data = self.gear_message.encode({'GEAR_SHIFTER': 0, 'GEAR': gear_data})
        message = can.Message(arbitration_id=self.gear_message.frame_id, data=encoded_gear_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    def log_blinker_data(self, left_blinker, right_blinker):
        encoded_blinker_data = self.blinker_message.encode({'DRIVERS_DOOR_OPEN': 0, 'MAIN_ON': 0,'RIGHT_BLINKER': right_blinker,'LEFT_BLINKER': left_blinker, 'CMBS_STATES': 0})
        message = can.Message(arbitration_id=self.blinker_message.frame_id, data=encoded_blinker_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    def log_headlight_data(self, low_beam, high_beam):
        headlight_data = 1 if (low_beam or high_beam) else 0
        encoded_headlight_data = self.headlight_message.encode({'AUTO_HEADLIGHTS': 0, 'HIGH_BEAM_HOLD': 0, 'HIGH_BEAM_FLASH': 0, 'HEADLIGHTS_ON': headlight_data, 'WIPER_SWITCH': 0})
        message = can.Message(arbitration_id=self.headlight_message.frame_id, data=encoded_headlight_data)
        # line = str(message) 
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    def log_beam_data(self, low_beam, high_beam, park_lights):
        encoded_beam_data = self.beam_message.encode({'WIPERS': 0, 'LOW_BEAMS': low_beam, 'HIGH_BEAMS': high_beam, 'PARK_LIGHTS': park_lights})
        message = can.Message(arbitration_id=self.beam_message.frame_id, data=encoded_beam_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    def log_handbrake_data(self, handbrake):
        handbrake_data = 1 if (handbrake == True) else 0
        encoded_handbrake_data = self.handbrake_message.encode({'ESP_DISABLED': 0, 'USER_BRAKE': handbrake_data, 'BRAKE_HOLD_ACTIVE': 0, 'BRAKE_HOLD_ENABLED': 0})
        message = can.Message(arbitration_id=self.handbrake_message.frame_id, data=encoded_handbrake_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    def log_speed_data(self, speed):
        encoded_speed_data = self.speed_message.encode({'CAR_SPEED': speed})
        message = can.Message(arbitration_id=self.speed_message.frame_id, data=encoded_speed_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    def inject_dos(self):
        message = can.Message(arbitration_id=0x00000000, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)

    def start_frame(self):
        # To do DoS with different arbitration IDs, you can modify the arbitration_id and data fields of the message. 
        message = can.Message(arbitration_id=0x1FFFFFFF, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)

    def inject_data(self, can_id, can_data):
        message = can.Message(arbitration_id=can_id, data=can_data, is_extended_id=True)
        self.bus.send(message)

    def log_dummy_data_1(self, queue_170):
        data=[0x00] * 8
        if len(queue_170) == 0:
            data[0] = random.randint(0, 255)
        else:
            data = queue_170.popleft()
        message = can.Message(arbitration_id=0x00000170, data=data, is_extended_id=True)
        self.bus.send(message)

    def log_dummy_data_2(self, queue_202):
        data=[0x00] * 8
        if len(queue_202) == 0:
            data[1] = random.randint(0, 255)
        else:
            data = queue_202.popleft()
        message = can.Message(arbitration_id=0x00000202, data=data, is_extended_id=True)
        self.bus.send(message)
    
    def log_dummy_data_3(self, queue_18F):
        data=[0x00] * 8
        if len(queue_18F) == 0:
            data[2] = random.randint(0, 255)
        else:
            data = queue_18F.popleft()
        message = can.Message(arbitration_id=0x0000018f, data=data, is_extended_id=True)
        self.bus.send(message)

    def log_dummy_data_4(self):
        data=[0x00] * 8
        data[3] = random.randint(0, 255)
        message = can.Message(arbitration_id=0x00000430, data=data, is_extended_id=True)
        self.bus.send(message)
    
    def log_dummy_data_5(self):
        data=[0x00] * 8
        data[4] = random.randint(0, 255)
        message = can.Message(arbitration_id=0x000001f1, data=data, is_extended_id=True)
        self.bus.send(message)
    
    def log_dummy_data_6(self):
        data=[0x00] * 8
        data[5] = random.randint(0, 255)
        message = can.Message(arbitration_id=0x000004b1, data=data, is_extended_id=True)
        self.bus.send(message)

    def log_data(self, index, time_diff_tick, steer_data, throttle_data, brake_data, gear_data,
             manual_gear_shift, left_blinker, right_blinker, low_beam, high_beam,
             park_lights, handbrake, speed_data, jitter_array, queue_170, queue_202, queue_18F):

        message_specs = [
            ("steer", lambda: self.log_steer_data(steer_data), 0.0),           # every tick
            ("throttle_brake", lambda: self.log_throttle_brake_data(throttle_data, brake_data), 0.0),
            ("speed", lambda: self.log_speed_data(speed_data), 0.0),
            ("gear", lambda: self.log_gear_data(gear_data, manual_gear_shift), 0.5),
            ("blinker", lambda: self.log_blinker_data(left_blinker, right_blinker), 1.0),
            ("headlight", lambda: self.log_headlight_data(low_beam, high_beam), 1.0),
            ("beam", lambda: self.log_beam_data(low_beam, high_beam, park_lights), 1.0),
            ("handbrake", lambda: self.log_handbrake_data(handbrake), 1.0),
            ("dummy_1", lambda: self.log_dummy_data_1(queue_170), 0.0),  # every tick
            ("dummy_2", lambda: self.log_dummy_data_2(queue_202), 0.25),
            ("dummy_3", lambda: self.log_dummy_data_3(queue_18F), 0.5),
            # ("dummy_4", lambda: self.log_dummy_data_4(), 0.75),
            # ("dummy_5", lambda: self.log_dummy_data_5(), 1.0),
            # ("dummy_6", lambda: self.log_dummy_data_6(), 2.0),
            # Example: Add a new message here with 500ms periodicity:
            # ("my_new_msg", lambda: self.log_new_data(data), 0.5),
        ]

        num_msgs = len(message_specs)

        def is_eligible(period):
            if period == 0.0:
                return True
            ticks_per_period = int(period / time_diff_tick) if time_diff_tick > 0 else 1
            return index % ticks_per_period == 0

        jitter_chunk = jitter_array[index * num_msgs : (index + 1) * num_msgs]

        eligible_actions = []
        for i, (name, log_fn, period) in enumerate(message_specs):
            if is_eligible(period):
                jitter = jitter_chunk[i]
                eligible_actions.append((jitter, log_fn))

        sorted_actions = sorted(eligible_actions, key=lambda x: x[0])
        for _, action in sorted_actions:
            action()
            add_delay(0.000200) #150 microseconds

    def parse_logs(self, steer_msg, throttle_brake_msg):
        steer = 0.0
        throttle, brake = 0.0, 0.0
        gear = 0

        # Handle steer message (can be None)
        if steer_msg is not None:
            can_id = steer_msg.arbitration_id
            if can_id == self.steer_message.frame_id:
                steer_data = self.steer_message.decode(steer_msg.data)
                steer = steer_data['STEER_ANGLE']

        # Handle throttle/brake message (can be None)
        if throttle_brake_msg is not None:
            can_id = throttle_brake_msg.arbitration_id
            if can_id == self.throttle_brake_message.frame_id:
                throttle_brake_data = self.throttle_brake_message.decode(throttle_brake_msg.data)
                throttle = throttle_brake_data['PEDAL_GAS']
                brake = throttle_brake_data['BRAKE_PRESSED']

        return throttle, steer, brake, False, False, False, gear
    
    def convert_to_CAN_msg(self, line):
        # print(line)
        parts = line.strip().split()
        timestamp_str = parts[0]
        can_id = parts[2]
        can_data = line.split("[8]")[1].replace(" ", "")

        can_msg = can.Message(
            arbitration_id=int(can_id, 16),
            data=bytearray.fromhex(can_data)
        )

        # Extract timestamp from "(000.123456)"
        if timestamp_str.startswith("(") and timestamp_str.endswith(")"):
            can_msg.timestamp = float(timestamp_str[1:-1])
        else:
            can_msg.timestamp = 0.0

        return can_msg

    def __del__(self):
        self.bus.shutdown()
        pass

def add_delay(delay):
    current_time = timeit.default_timer()
    delay_time = current_time + delay
    while current_time < delay_time:
        current_time = timeit.default_timer()

def generate_jitter_array(min_jitter, max_jitter, size):
    return [random.uniform(min_jitter, max_jitter) for _ in range(size)]

def get_status(vehicle):
    # Initialize variables dynamically
    current_lights = vehicle.get_light_state()
    light_status = {
        "left_blinker_set": int(bool(current_lights & carla.VehicleLightState.LeftBlinker)),
        "right_blinker_set": int(bool(current_lights & carla.VehicleLightState.RightBlinker)),
        "low_beam_set": int(bool(current_lights & carla.VehicleLightState.LowBeam)),
        "high_beam_set": int(bool(current_lights & carla.VehicleLightState.HighBeam)),
        "park_lights_set": int(bool(current_lights & carla.VehicleLightState.Position))
    }

    velocity = vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

    # print(f'Speed: {speed:.2f} km/h')

    return light_status, speed

def generate_dos_indices(lower, upper, count):
    if count > (upper - lower + 1):
        raise ValueError("Count exceeds the number of available indices in the range.")
    return sorted(random.sample(range(lower, upper + 1), count))

def generate_chunkwise_dos_indices(total_dos_msgs, num_ranges=5, range_size=10000, max_index=200000):
    # Step 1: Compute all possible non-overlapping ranges
    possible_ranges = [(start, start + range_size) for start in range(0, max_index, range_size)]
    
    if len(possible_ranges) < num_ranges:
        raise ValueError("Not enough non-overlapping ranges available.")

    # Step 2: Randomly select non-overlapping index ranges
    selected_ranges = random.sample(possible_ranges, num_ranges)

    # Step 3: Collect eligible indices from the selected ranges
    eligible_indices = []
    for start, end in selected_ranges:
        eligible_indices.extend(range(start, min(end, max_index)))

    # Step 4: Randomly sample the required number of DoS indices
    if total_dos_msgs > len(eligible_indices):
        raise ValueError("Requested DoS messages exceed available indices in selected ranges.")

    dos_indices = sorted(random.sample(eligible_indices, total_dos_msgs))
    return dos_indices

def modify_vehicle_physics(actor):
    try:
        physics_control = actor.get_physics_control()

        # Enable more realistic wheel collisions
        physics_control.use_sweep_wheel_collision = True

        # Adjust the torque curve: (RPM_ratio, Torque)
        # RPM ratio is from 0.0 to 1.0
        physics_control.torque_curve = [
            (0.0, 500.0),   # Stronger torque at low RPM
            (0.5, 1000.0),  # Peak torque at mid RPM
            (1.0, 800.0)    # Slight drop-off at high RPM
        ]

        # Increase the max RPM to allow higher engine speed
        physics_control.max_rpm = 5000

        # Lower the drag to reduce resistance (default is often ~0.35–0.45)
        physics_control.drag_coefficient = 0.25

        # Enable automatic gear shifting
        physics_control.use_gear_autobox = True

        # Gear ratios for better acceleration and top speed
        physics_control.gear_ratios = [4.0, 3.0, 2.0, 1.5, 1.0, 0.9, 0.8]

        # Optional: faster gear switching
        physics_control.gear_switch_time = 0.15

        # Apply all changes
        actor.apply_physics_control(physics_control)

    except Exception as e:
        print(f"[WARN] Failed to modify physics for actor {actor.id}: {e}")

def read_jitter_array(file_path):
    jitter_array = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Check if the line is not empty
                    jitter_value = float(line)
                    jitter_array.append(jitter_value)
    except FileNotFoundError:
        print(f"[ERROR] Jitter array file not found: {file_path}")
    except ValueError as e:
        print(f"[ERROR] Invalid value in jitter array file: {e}")
    return jitter_array

# def convert_can_to_control_data(file_path):
#     control_obj = []
#     can_logger = CAN_Data_Logger()
    
#     current_group = {'14A': None, '17C': None}

#     def add_control_obj():
#         steer_msg = current_group['14A']
#         throttle_msg = current_group['17C']
        
#         throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(steer_msg, throttle_msg)
        
#         control_obj.append(carla.VehicleControl(
#             throttle=throttle,
#             steer=steer,
#             brake=brake,
#             gear=gear
#         ))

#     with open(file_path, 'r') as file:
#         for line in file:
#             msg = can_logger.convert_to_CAN_msg(line)
#             can_id = msg.arbitration_id

#             if can_id == 0x14A:
#                 current_group['14A'] = msg

#             elif can_id == 0x17C:
#                 current_group['17C'] = msg

#             # If all two messages are present, process
#             if all(current_group.values()):
#                 add_control_obj()
#                 current_group = {'14A': None, '17C': None}

#     return control_obj

# def convert_can_to_control_data(file_path):
#     control_obj = []
#     can_logger = CAN_Data_Logger()

#     current_group = {'14A': None, '17C': None}

#     # Initialize queues for specific CAN IDs
#     queue_170 = deque()
#     queue_202 = deque()
#     queue_18F = deque()

#     # Helper function to add parsed VehicleControl object
#     def add_control_obj():
#         steer_msg = current_group['14A']
#         throttle_msg = current_group['17C']
        
#         throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(steer_msg, throttle_msg)
        
#         control_obj.append(carla.VehicleControl(
#             throttle=throttle,
#             steer=steer,
#             brake=brake,
#             gear=gear
#         ))

#     with open(file_path, 'r') as file:
#         for line in file:
#             # Extract CAN ID in hex string format
#             parts = line.strip().split()
#             if len(parts) >= 6:
#                 try:
#                     hex_id = parts[2]
#                     data_bytes = parts[4:12]  # List of 8 data bytes
#                     data_ints = [int(byte, 16) for byte in data_bytes]  # Convert strings to integers (base 16)

#                     if hex_id == '00000170':
#                         queue_170.append(data_ints)
#                     elif hex_id == '00000202':
#                         queue_202.append(data_ints)
#                     elif hex_id == '0000018F':
#                         queue_18F.append(data_ints)
#                 except IndexError:
#                     continue  # Malformed line; skip

#             # Convert to CAN message object
#             try:
#                 msg = can_logger.convert_to_CAN_msg(line)
#                 can_id = msg.arbitration_id

#                 if can_id == 0x14A:
#                     current_group['14A'] = msg
#                 elif can_id == 0x17C:
#                     current_group['17C'] = msg

#                 # If both messages available, parse and reset
#                 if all(current_group.values()):
#                     add_control_obj()
#                     current_group = {'14A': None, '17C': None}
#             except Exception:
#                 continue  # Ignore lines that can't be parsed

#     # You now have:
#     # - control_obj: list of VehicleControl objects
#     # - queue_170, queue_202, queue_18F: CAN data queues

#     return control_obj, queue_170, queue_202, queue_18F

def convert_can_to_control_data(file_path):
    control_obj = []
    control_timestamps = []
    can_logger = CAN_Data_Logger()

    current_group = {'14A': None, '17C': None}
    last_control = None  # Keep track of last control values

    queue_170 = deque()
    queue_202 = deque()
    queue_18F = deque()

    # Helper function to add parsed VehicleControl object
    def add_control_obj(steer_msg, throttle_msg):
        try:
            if steer_msg and throttle_msg:
                throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(steer_msg, throttle_msg)
            elif steer_msg and last_control:
                throttle, brake, gear = last_control.throttle, last_control.brake, last_control.gear
                _, steer, _, _, _, _, _ = can_logger.parse_logs(steer_msg, None)
            elif throttle_msg and last_control:
                steer = last_control.steer
                throttle, _, brake, _, _, _, gear = can_logger.parse_logs(None, throttle_msg)
            else:
                return None  # Can't form control without at least steer or throttle and fallback

            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                gear=gear
            )
            control_obj.append(control)
            return control
        except Exception:
            return None  # Ignore parse errors

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    hex_id = parts[2]
                    data_bytes = parts[4:12]
                    data_ints = [int(byte, 16) for byte in data_bytes]

                    if hex_id == '00000170':
                        queue_170.append(data_ints)
                    elif hex_id == '00000202':
                        queue_202.append(data_ints)
                    elif hex_id == '0000018F':
                        queue_18F.append(data_ints)
                except IndexError:
                    continue  # Malformed line; skip

            try:
                msg = can_logger.convert_to_CAN_msg(line)
                can_id = msg.arbitration_id

                if can_id == 0x14A:
                    # Steer message
                    if current_group['14A']:
                        control_timestamps.append(current_group['14A'].timestamp)  # Store timestamp for control
                        control = add_control_obj(current_group['14A'], current_group['17C'])
                        if control:
                            last_control = control
                    current_group['14A'] = msg

                elif can_id == 0x17C:
                    # Throttle message
                    if current_group['17C']:
                        control_timestamps.append(current_group['17C'].timestamp)  # Store timestamp for control
                        control = add_control_obj(current_group['14A'], current_group['17C'])
                        if control:
                            last_control = control
                    current_group['17C'] = msg

                # If both messages now present, process and reset
                if current_group['14A'] and current_group['17C']:
                    control = add_control_obj(current_group['14A'], current_group['17C'])
                    if control:
                        last_control = control
                    current_group = {'14A': None, '17C': None}

            except Exception:
                continue  # Skip unparseable lines

    return control_obj, queue_170, queue_202, queue_18F, control_timestamps


def extract_control_data(file_path):
    control_obj = []
    pattern = re.compile(r"throttle=([\d\.\-]+), steer=([\d\.\-]+), brake=([\d\.\-]+).*gear=(\d+)")
    
    # Read the log file and extract control data
    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                throttle, steer, brake, gear = match.groups()
                control_obj.append(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake), gear=int(gear)))
    
    return control_obj

def parse_can_log(file_path):
    # Define the regex pattern
    pattern = re.compile(r"\(([\d\.]+)\)\s+vcan0\s+([0-9A-Fa-f]{8})\s+\[\d\]\s+((?:[0-9A-Fa-f]{2}\s+){7}[0-9A-Fa-f]{2})")

    parsed_data = []

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp = float(match.group(1))
                can_id = int(match.group(2), 16)  # Convert to integer
                data = [int(byte_str, 16) for byte_str in match.group(3).strip().split()]
                parsed_data.append({
                    'timestamp': timestamp,
                    'id': can_id,
                    'data': data
                })

    return parsed_data

def replay_dos_data_two_car():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the map
    world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)
    # world = client.get_world()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicle
    # car_model_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
    # car_model = car_model_list[2]
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    source = spawn_points[11]
    vehicle_2 = world.try_spawn_actor(blueprint, source)
    # modify_vehicle_physics(vehicle_2)
    world.wait_for_tick()
    source.location.y -= 25
    vehicle_1 = world.try_spawn_actor(blueprint, source)
    # modify_vehicle_physics(vehicle_1)
    world.wait_for_tick()

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.008
    world.apply_settings(settings)

    opt_dict = {'follow_speed_limits': False, 'ignore_traffic_lights': True, 'ignore_stop_signs': True, 'ignore_vehicles': True}

    # agent_1 = BasicAgent(vehicle_1, target_speed=10, opt_dict=opt_dict)
    # agent_2 = BasicAgent(vehicle_2, target_speed=10, opt_dict=opt_dict)
    agent_1 = BehaviorAgent(vehicle_1, behavior='normal', opt_dict=opt_dict)
    agent_2 = BehaviorAgent(vehicle_2, behavior='normal', opt_dict=opt_dict)
   
    new_index = 20
    destination = spawn_points[new_index].location
    agent_1.set_destination(destination)
    agent_2.set_destination(destination)

    # Open log files
    vehicle_location_writer_1 = open('./Logs_Replay/gen_coord_1.log', 'w')
    vehicle_control_writer_1 = open("./Logs_Replay/gen_control_obj_1.log", "w")

    vehicle_location_writer_2 = open('./Logs_Replay/gen_coord_2.log', 'w')
    vehicle_control_writer_2 = open("./Logs_Replay/gen_control_obj_2.log", "w")

    timestamp_writer = open('./Logs_Replay/gen_timestamps.log', 'w')
    dos_timestamp_writer = open('./Logs_Replay/dos_timestamp.log', 'w')

    timestamp_reader = open('./Logs_25.5_1/gen_vehicle_control_time.log', 'r')
    
    # Initialize lists to store data
    vehicle_control_obj_1, queue_170, queue_202, queue_18F, control_timestamps  = convert_can_to_control_data('./Logs_25.5_1/can_data_logs.log')
    # vehicle_control_obj_1 = extract_control_data('./Logs_25.5_1/gen_control_obj_1.log')
    vehicle_control_obj_2 = extract_control_data('./Logs_25.5_1/gen_control_obj_2.log')
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []
    dos_timestamp = []

    # Read timestamps from the generated log file
    gen_timestamps = [float(line.strip()) for line in timestamp_reader if line.strip()]
    gen_timestamps.extend(control_timestamps)  # Append control timestamps to the generated timestamps
    gen_timestamps.sort()  # Sort all timestamps

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    time_diff_tick = 0.006

    sim_time_sec = 36 #Duration of simulation in seconds
    total_iterations = int(sim_time_sec / time_diff_tick)

    jitter_array = read_jitter_array('./Logs_25.5_1/jitter_array.log') 

    dos_mode = False
    count_dos = 0
    num_dos_msgs = 100 # Number of DoS messages to inject
    print(f"Number of DoS messages to inject: {num_dos_msgs}")
    skipped_control_objects = queue.Queue()

    # Start simulation
    start_time = timeit.default_timer()
    can_handler.start_frame()
    try:
        for i in range(len(vehicle_control_obj_1)):  

            # Get current time
            # current_time = timeit.default_timer() - start_time

            # while current_time < gen_timestamps[i]:
            #     current_time = timeit.default_timer() - start_time

            # timestamps.append(current_time)

            world.tick()

            if count_dos == num_dos_msgs:
                while not skipped_control_objects.empty():
                    # print("Skipping control object")
                    control = skipped_control_objects.get()
                    vehicle_1.apply_control(control)

                    light_status, speed = get_status(vehicle_1)

                    can_handler.log_data(
                            i,
                            time_diff_tick,
                            control.steer,
                            control.throttle,
                            control.brake,
                            control.gear,
                            control.manual_gear_shift,
                            light_status["left_blinker_set"],
                            light_status["right_blinker_set"],
                            light_status["low_beam_set"],
                            light_status["high_beam_set"],
                            light_status["park_lights_set"],
                            control.hand_brake,
                            speed,
                            jitter_array,
                            queue_170, queue_202, queue_18F
                        )

                    # vehicle_control_obj_1.append(control)

                count_dos = 0

            # # Get current control state
            control_1 = vehicle_control_obj_1[i]
            control_2 = agent_2.run_step()

            current_location_1 = vehicle_1.get_location()
            current_location_2 = vehicle_2.get_location()

            #Application of benign control
            vehicle_2.apply_control(control_2)

            current_time = timeit.default_timer() - start_time

            while current_time < gen_timestamps[i]:
                current_time = timeit.default_timer() - start_time

            timestamps.append(current_time)

            # DoS Attack in a specific time range
            if dos_mode or (current_time >= 25.5 and current_time <= 25.506):
                dos_mode = True
                count_dos += 1
                # Check for number of benign timestamp to attack
                if count_dos >= num_dos_msgs:
                    dos_mode = False

                # Inject DoS message
                can_handler.inject_dos()
                # print(f"Injecting DoS message at time {current_time:.6f} seconds")

                skipped_control_objects.put(control_1)
                dos_timestamp.append(current_time)
            else:
                #Application of benign control
                vehicle_1.apply_control(control_1)
                light_status, speed = get_status(vehicle_1)

                can_handler.log_data(
                        i,
                        time_diff_tick,
                        control_1.steer,
                        control_1.throttle,
                        control_1.brake,
                        control_1.gear,
                        control_1.manual_gear_shift,
                        light_status["left_blinker_set"],
                        light_status["right_blinker_set"],
                        light_status["low_beam_set"],
                        light_status["high_beam_set"],
                        light_status["park_lights_set"],
                        control_1.hand_brake,
                        speed,
                        jitter_array,
                        queue_170, queue_202, queue_18F
                    )
                
                # vehicle_control_obj_1.append(control_1)
           
            # print(speed)

            #Log Vehicle Control Data  
            # vehicle_control_obj_2.append(control_2)

            # Logging vehicle location data
            vehicle_location_1.append(current_location_1)
            vehicle_location_2.append(current_location_2)

    finally:
        
        print("Terminating simulation...")
        print()

        for control in vehicle_control_obj_1:
            vehicle_control_writer_1.write(str(control) + "\n")
        vehicle_control_writer_1.close()

        for control in vehicle_control_obj_2:
            vehicle_control_writer_2.write(str(control) + "\n")
        vehicle_control_writer_2.close()
        
        for ele in timestamps:
            if (ele < 0.0001):
                timestamp_writer.write("00.000000" + '\n')
            elif (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        for location in vehicle_location_1:
            vehicle_location_writer_1.write(str(location) + "\n")
        vehicle_location_writer_1.close()

        for location in vehicle_location_2:
            vehicle_location_writer_2.write(str(location) + "\n")
        vehicle_location_writer_2.close()

        for ele in dos_timestamp:
            if (ele < 10):
                dos_timestamp_writer.write("0" + str(ele)[:11] + '\n')
            else:
                dos_timestamp_writer.write(str(ele)[:12] + '\n')
        dos_timestamp_writer.close()

        plot_vc_time(vehicle_control_obj_1, timestamps, "./Graphs_Replay/plot_throttle_time_1.png", "./Graphs_Replay/plot_steer_time_1.png", "./Graphs_Replay/plot_brake_time_1.png")
        plot_diff(timestamps, "./Graphs_Replay/plot_timestamp_diff_gen.png")
        plot_euclid_diff_by_index_only(vehicle_location_1, vehicle_location_2,'./Logs', './Graphs_Replay/plot_euclid_diff_gen_index.png', 0, total_iterations)
        plot_gen_vs_rep_paths_from_files("./Logs/gen_coord_1.log", "./Logs_Replay/gen_coord_1.log", "./Graphs_Replay/benign_vs_attack_path.png", 0, total_iterations)
        plot_benign_timeline_near_dos(timestamps, dos_timestamp, './Graphs_Replay/plot_dos_timeline.png')

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")

def replay_spoof_data_two_car():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the map
    world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)
    # world = client.get_world()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicle
    # car_model_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
    # car_model = car_model_list[2]
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    source = spawn_points[11]
    vehicle_2 = world.try_spawn_actor(blueprint, source)
    # modify_vehicle_physics(vehicle_2)
    world.wait_for_tick()
    source.location.y -= 25
    vehicle_1 = world.try_spawn_actor(blueprint, source)
    # modify_vehicle_physics(vehicle_1)
    world.wait_for_tick()

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.008
    world.apply_settings(settings)

    opt_dict = {'follow_speed_limits': False, 'ignore_traffic_lights': True, 'ignore_stop_signs': True, 'ignore_vehicles': True}

    # agent_1 = BasicAgent(vehicle_1, target_speed=60, opt_dict=opt_dict)
    # agent_2 = BasicAgent(vehicle_2, target_speed=60, opt_dict=opt_dict)
    agent_1 = BehaviorAgent(vehicle_1, behavior='normal', opt_dict=opt_dict)
    agent_2 = BehaviorAgent(vehicle_2, behavior='normal', opt_dict=opt_dict)
   
    new_index = 20
    destination = spawn_points[new_index].location
    agent_1.set_destination(destination)
    agent_2.set_destination(destination)

    # Open log files
    vehicle_location_writer_1 = open('./Logs_Replay/gen_coord_1.log', 'w')
    vehicle_control_writer_1 = open("./Logs_Replay/gen_control_obj_1.log", "w")

    vehicle_location_writer_2 = open('./Logs_Replay/gen_coord_2.log', 'w')
    vehicle_control_writer_2 = open("./Logs_Replay/gen_control_obj_2.log", "w")

    timestamp_writer = open('./Logs_Replay/gen_timestamps.log', 'w')
    spoof_timestamp_writer = open('./Logs_Replay/spoof_timestamp.log', 'w')

    timestamp_reader = open('./Logs_36.0_1/gen_vehicle_control_time.log', 'r')
    spoof_timestamp_reader = open('./Logs_36.0_1/spoof_timestamp.log', 'r')

    # Initialize lists to store data
    vehicle_control_obj_1 = []
    vehicle_control_obj_2 = []
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []
    spoof_timestamp = []
    # temp_control_obj = []

    stored_control_obj_1, queue_170, queue_202, queue_18F, control_timestamps  = convert_can_to_control_data('./Logs_36.0_1/can_data_logs.log')
    # stored_control_obj_1 = extract_control_data('./Logs_36.0_1/gen_control_obj_1.log')
    stored_control_obj_2 = extract_control_data('./Logs_36.0_1/gen_control_obj_2.log')

    # Read timestamps from the generated log file
    gen_timestamps = [float(line.strip()) for line in timestamp_reader if line.strip()]
    gen_timestamps.extend(control_timestamps)  # Append control timestamps to the generated timestamps
    gen_timestamps.sort()  # Sort all timestamps
    # print(gen_timestamps)

    gen_spoof_timestamps = [float(line.strip()) for line in spoof_timestamp_reader if line.strip()]

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    time_diff_tick = 0.006

    sim_time_sec = 42.0 # Duration of simulation in seconds
    total_iterations = int(sim_time_sec / time_diff_tick)

    jitter_array = read_jitter_array('./Logs_36.0_1/jitter_array.log')

    spoof_mode = False
    count_spoof = 0
    num_spoof_msgs = 50
    print("Number spoof: ", num_spoof_msgs)
    spoof_delay = 0.0025 # Delay for spoofed control in seconds
    print("Amount of Delay: ", spoof_delay) 

    i = 0
    j = 0

    # Start simulation
    start_time = timeit.default_timer()
    can_handler.start_frame()
    try:
        while i < len(stored_control_obj_1):  

            # # Get current time
            # current_time = timeit.default_timer() - start_time

            # # print(str(current_time))
            # timestamps.append(current_time)

            world.tick()

            # Get current control state
            # print(f"Current index: {i}, j: {j}, control object: {str(stored_control_obj_1[i])}")
            control_1 = stored_control_obj_1[i]
            control_2 = agent_2.run_step()

            i += 1

            current_location_1 = vehicle_1.get_location()
            current_location_2 = vehicle_2.get_location()

            current_time = timeit.default_timer() - start_time
            # print(f"Current time: {current_time:.6f} seconds, Generated timestamp: {gen_timestamps[j]:.6f} seconds")
            while current_time < gen_timestamps[j]:
                current_time = timeit.default_timer() - start_time

            timestamps.append(current_time)

            #Application of benign control
            vehicle_1.apply_control(control_1)
            vehicle_2.apply_control(control_2)

            light_status, speed = get_status(vehicle_1)

            can_handler.log_data(
                    i,
                    time_diff_tick,
                    control_1.steer,
                    control_1.throttle,
                    control_1.brake,
                    control_1.gear,
                    control_1.manual_gear_shift,
                    light_status["left_blinker_set"],
                    light_status["right_blinker_set"],
                    light_status["low_beam_set"],
                    light_status["high_beam_set"],
                    light_status["park_lights_set"],
                    control_1.hand_brake,
                    speed,
                    jitter_array,
                    queue_170, queue_202, queue_18F
                )

            #Log Vehicle Control Data  
            # print(i+1, control_1)
            vehicle_control_obj_1.append(control_1)
            vehicle_control_obj_2.append(control_2)

            # Spoofing Attack in a specific time range
            if spoof_mode or (current_time >= 36.0 and current_time <= 36.006):
                spoof_mode = True
                count_spoof += 1
                # Check for number of benign timestamp to attack
                if count_spoof >= num_spoof_msgs:
                    spoof_mode = False

                # Add delay to the spoofed control
                # delay_time = current_time + spoof_delay
                # current_timestamp = timeit.default_timer()-start_time
                # while current_timestamp < delay_time:
                #     current_timestamp = timeit.default_timer()-start_time

                # Apply the spoofed control to the first vehicle in the current tick
                count = 0
                # control_1.steer = 1.0
                
                # print(f"Current index: {i}, j: {j}, control object: {str(stored_control_obj_1[i])}")
                control = stored_control_obj_1[i]
                i += 1
                while count < 1:
                    current_timestamp = timeit.default_timer()-start_time
                    while current_timestamp < gen_spoof_timestamps[count_spoof-1]:
                        current_timestamp = timeit.default_timer() - start_time

                    spoof_timestamp.append(current_timestamp)

                    vehicle_1.apply_control(control)
                    light_status, speed = get_status(vehicle_1)

                    can_handler.log_data(
                            i,
                            time_diff_tick,
                            control.steer,
                            control.throttle,
                            control.brake,
                            control.gear,
                            control.manual_gear_shift,
                            light_status["left_blinker_set"],
                            light_status["right_blinker_set"],
                            light_status["low_beam_set"],
                            light_status["high_beam_set"],
                            light_status["park_lights_set"],
                            control.hand_brake,
                            speed,
                            jitter_array,
                            queue_170, queue_202, queue_18F
                        )
                    # print(i+1, control_1)
                    vehicle_control_obj_1.append(control)

                    count += 1

            # Logging vehicle location data
            vehicle_location_1.append(current_location_1)
            vehicle_location_2.append(current_location_2)

            # Increment iteration
            j += 1

    finally:
        
        print("Terminating simulation...")
        print()

        for control in vehicle_control_obj_1:
            vehicle_control_writer_1.write(str(control) + "\n")
        vehicle_control_writer_1.close()

        for control in vehicle_control_obj_2:
            vehicle_control_writer_2.write(str(control) + "\n")
        vehicle_control_writer_2.close()
        
        for ele in timestamps:
            if (ele < 0.000001):
                timestamp_writer.write("00.000000" + '\n')
            elif (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        for location in vehicle_location_1:
            vehicle_location_writer_1.write(str(location) + "\n")
        vehicle_location_writer_1.close()

        for location in vehicle_location_2:
            vehicle_location_writer_2.write(str(location) + "\n")
        vehicle_location_writer_2.close()

        for ele in spoof_timestamp:
            if (ele < 10):
                spoof_timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                spoof_timestamp_writer.write(str(ele)[:9] + '\n')
        spoof_timestamp_writer.close()

        plot_vc_time(vehicle_control_obj_1, timestamps, "./Graphs_Replay/plot_throttle_time_1.png", "./Graphs_Replay/plot_steer_time_1.png", "./Graphs_Replay/plot_brake_time_1.png")
        plot_euclid_diff_by_index_only(vehicle_location_1, vehicle_location_2,'./Logs', './Graphs_Replay/plot_euclid_diff_gen_index.png', 0, total_iterations)
        plot_gen_vs_rep_paths_from_files("./Logs/gen_coord_1.log", "./Logs_Replay/gen_coord_1.log", "./Graphs_Replay/benign_vs_attack_path.png", 0, total_iterations)
        plot_spoof_timeline(timestamps, spoof_timestamp, './Graphs_Replay/plot_spoof_timeline_combined.png')

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")

if __name__ == '__main__':
    print("Starting generation...")
    # replay_dos_data_two_car()
    replay_spoof_data_two_car()