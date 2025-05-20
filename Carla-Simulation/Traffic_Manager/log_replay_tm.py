import glob
import queue
import sys
import timeit

import pygame
import carla
import time
import re
import os
import can
import cantools
from plot_graphs import plot_diff, plot_euclid_diff_by_index_only, plot_euclid_diff_single_function, plot_gen_vs_rep_paths_from_files, plot_vc_index, plot_vc_time

"""
    This script is designed to replay vehicle can data in a CARLA simulation environment.
"""

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


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

    def convert_to_CAN_msg(self, line):
        can_id = line.split()[2]
        can_data = line.split("[8]")[1].replace(" ", "")
        can_msg = can.Message(arbitration_id=int(can_id, 16), data=bytearray.fromhex(can_data))
        return can_msg

    def parse_logs(self, steer_msg, throttle_brake_msg,gear_msg):
        steer = 0.0
        can_id = steer_msg.arbitration_id

        if can_id == self.steer_message.frame_id:
            steer_data = self.steer_message.decode(steer_msg.data)
            steer = steer_data['STEER_ANGLE']
        
        throttle, brake = 0.0, 0.0
        can_id = throttle_brake_msg.arbitration_id

        if can_id == self.throttle_brake_message.frame_id:
            throttle_brake_data = self.throttle_brake_message.decode(throttle_brake_msg.data)
            throttle = throttle_brake_data['PEDAL_GAS']
            brake = throttle_brake_data['BRAKE_PRESSED']

        gear = 0
        can_id = gear_msg.arbitration_id

        if can_id == self.gear_message.frame_id:
            gear_data = self.gear_message.decode(gear_msg.data)
            gear = gear_data['GEAR']
        
        return throttle, steer, brake, False, False, False, gear
    
    def log_steer_data(self, steer_data):
        encoded_steer_data = self.steer_message.encode({'STEER_ANGLE': steer_data})
        message = can.Message(arbitration_id=self.steer_message.frame_id, data=encoded_steer_data)
        line = str(message)
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)

    def log_throttle_brake_data(self, throttle_data, brake_data):
        encoded_throttle_brake_data = self.throttle_brake_message.encode({'PEDAL_GAS': throttle_data, 'ENGINE_RPM': 0, 'GAS_PRESSED': 0, 
                                                   'ACC_STATUS': 0, 'BOH_17C': 0, 'BRAKE_SWITCH': 0, 
                                                   'BOH2_17C': 0, 'BRAKE_PRESSED': brake_data})
        message = can.Message(arbitration_id=self.throttle_brake_message.frame_id, data=encoded_throttle_brake_data)
        line = str(message)
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)
       
    def log_gear_data(self, gear_data, manual_gear_shift):
        manual_gear_shift = 1 if (manual_gear_shift == True) else 0   
        if manual_gear_shift:    #1 for manual and 0 for auto transmission
            encoded_gear_data = self.gear_message.encode({'GEAR_SHIFTER': 1, 'GEAR': gear_data})
        else:
            encoded_gear_data = self.gear_message.encode({'GEAR_SHIFTER': 0, 'GEAR': gear_data})
        message = can.Message(arbitration_id=self.gear_message.frame_id, data=encoded_gear_data)
        line = str(message)
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)

    def log_blinker_data(self, left_blinker, right_blinker):
        encoded_blinker_data = self.blinker_message.encode({'DRIVERS_DOOR_OPEN': 0, 'MAIN_ON': 0,'RIGHT_BLINKER': right_blinker,'LEFT_BLINKER': left_blinker, 'CMBS_STATES': 0})
        message = can.Message(arbitration_id=self.blinker_message.frame_id, data=encoded_blinker_data)
        line = str(message)
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)

    def log_headlight_data(self, low_beam, high_beam):
        headlight_data = 1 if (low_beam or high_beam) else 0
        encoded_headlight_data = self.headlight_message.encode({'AUTO_HEADLIGHTS': 0, 'HIGH_BEAM_HOLD': 0, 'HIGH_BEAM_FLASH': 0, 'HEADLIGHTS_ON': headlight_data, 'WIPER_SWITCH': 0})
        message = can.Message(arbitration_id=self.headlight_message.frame_id, data=encoded_headlight_data)
        line = str(message) 
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)

    def log_beam_data(self, low_beam, high_beam, park_lights):
        encoded_beam_data = self.beam_message.encode({'WIPERS': 0, 'LOW_BEAMS': low_beam, 'HIGH_BEAMS': high_beam, 'PARK_LIGHTS': park_lights})
        message = can.Message(arbitration_id=self.beam_message.frame_id, data=encoded_beam_data)
        line = str(message)
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)

    def log_handbrake_data(self, handbrake):
        handbrake_data = 1 if (handbrake == True) else 0
        encoded_handbrake_data = self.handbrake_message.encode({'ESP_DISABLED': 0, 'USER_BRAKE': handbrake_data, 'BRAKE_HOLD_ACTIVE': 0, 'BRAKE_HOLD_ENABLED': 0})
        message = can.Message(arbitration_id=self.handbrake_message.frame_id, data=encoded_handbrake_data)
        line = str(message)
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)

    def log_speed_data(self, speed):
        encoded_speed_data = self.speed_message.encode({'CAR_SPEED': speed})
        message = can.Message(arbitration_id=self.speed_message.frame_id, data=encoded_speed_data)
        line = str(message)
        match = re.search(self.pattern, line)
        id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        os.system(cmd)

    def log_data(self, steer_data, throttle_data, brake_data, gear_data, manual_gear_shift, left_blinker, right_blinker, low_beam, high_beam, park_lights, handbrake, speed_data):
        self.log_steer_data(steer_data)
        self.log_throttle_brake_data(throttle_data, brake_data)
        self.log_gear_data(gear_data, manual_gear_shift)
        self.log_blinker_data(left_blinker, right_blinker)
        self.log_headlight_data(low_beam, high_beam)
        self.log_beam_data(low_beam, high_beam, park_lights)
        self.log_handbrake_data(handbrake)
        self.log_speed_data(speed_data)

# def convert_can_to_control_data(file_path):
#     control_obj = []
#     can_logger = CAN_Data_Logger()
#     can_logs_array = []

#     # Read the log file and convert each line to a CAN message
#     with open(file_path, 'r') as file:
#         for line in file:
#             data = can_logger.convert_to_CAN_msg(line)
#             can_logs_array.append(data)
    
#     # Process the CAN messages in groups of 3 and extract control data
#     for i in range(0,len(can_logs_array),3):
#         throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(can_logs_array[i], can_logs_array[i + 1], can_logs_array[i + 2])
#         control_obj.append(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, gear=gear))
    
#     return control_obj

def convert_can_to_control_data(file_path):
    control_obj = []
    can_logger = CAN_Data_Logger()
    
    current_group = {'14A': None, '17C': None, '1A3': None}

    def add_control_obj():
        steer_msg = current_group['14A']
        throttle_msg = current_group['17C']
        brake_msg = current_group['1A3']
        
        if steer_msg:
            if throttle_msg and brake_msg:
                throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(steer_msg, throttle_msg, brake_msg)
            else:
                # Spoofed group (only 14A available), use defaults
                _, steer, _, _, _, _, gear = can_logger.parse_logs(steer_msg, steer_msg, steer_msg)
                throttle = 0.75
                brake = 0.0

            control_obj.append(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                gear=gear
            ))

    with open(file_path, 'r') as file:
        for line in file:
            msg = can_logger.convert_to_CAN_msg(line)
            can_id = msg.arbitration_id

            if can_id == 0x14A:
                if current_group['14A'] is not None:
                    # If we have a previous 14A message, process it
                    add_control_obj()
                    current_group = {'14A': None, '17C': None, '1A3': None}

                current_group['14A'] = msg

            elif can_id == 0x17C:
                current_group['17C'] = msg

            elif can_id == 0x1A3:
                current_group['1A3'] = msg

            # If all three messages are present, process
            if all(current_group.values()):
                add_control_obj()
                current_group = {'14A': None, '17C': None, '1A3': None}

    # Catch any remaining message (like stray 14A at the end)
    if current_group['14A']:
        add_control_obj()

    return control_obj

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

# def convert_can_to_control_data_dos(file_path):
#     control_obj = []
#     can_logger = CAN_Data_Logger()

#     current_group = {'14A': None, '17C': None, '1A3': None}
#     dos_indices = []
#     index_counter = 0  # Keep track of position in the control list

#     def add_control_obj():
#         nonlocal index_counter
#         steer_msg = current_group['14A']
#         throttle_msg = current_group['17C']
#         brake_msg = current_group['1A3']
        
#         if steer_msg:
#             if throttle_msg and brake_msg:
#                 throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(steer_msg, throttle_msg, brake_msg)
#             else:
#                 _, steer, _, _, _, _, gear = can_logger.parse_logs(steer_msg, steer_msg, steer_msg)
#                 throttle = 0.75
#                 brake = 0.0

#             control_obj.append(carla.VehicleControl(
#                 throttle=throttle,
#                 steer=steer,
#                 brake=brake,
#                 gear=gear
#             ))
#             index_counter += 1

#     with open(file_path, 'r') as file:
#         for line in file:
#             msg = can_logger.convert_to_CAN_msg(line)
#             can_id = msg.arbitration_id

#             if can_id == 0x000:
#                 dos_indices.append(index_counter)
#                 continue

#             if can_id == 0x14A:
#                 if current_group['14A'] is not None:
#                     add_control_obj()
#                     current_group = {'14A': None, '17C': None, '1A3': None}
#                 current_group['14A'] = msg
#             elif can_id == 0x17C:
#                 current_group['17C'] = msg
#             elif can_id == 0x1A3:
#                 current_group['1A3'] = msg

#             if all(current_group.values()):
#                 add_control_obj()
#                 current_group = {'14A': None, '17C': None, '1A3': None}

#     if current_group['14A']:
#         add_control_obj()

#     return control_obj, dos_indices

def convert_can_to_control_data_dos(file_path):
    control_obj = []
    can_logger = CAN_Data_Logger()

    current_group = {'14A': None, '17C': None, '1A3': None}
    dos_indices = []
    index_counter = 0  # Keep track of position in the control list

    def add_control_obj():
        nonlocal index_counter
        steer_msg = current_group['14A']
        throttle_msg = current_group['17C']
        brake_msg = current_group['1A3']
        
        if steer_msg:
            if throttle_msg and brake_msg:
                throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(steer_msg, throttle_msg, brake_msg)
            else:
                _, steer, _, _, _, _, gear = can_logger.parse_logs(steer_msg, steer_msg, steer_msg)
                throttle = 0.75
                brake = 0.0

            control_obj.append(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                gear=gear
            ))
            index_counter += 1

    with open(file_path, 'r') as file:
        for line in file:
            msg = can_logger.convert_to_CAN_msg(line)
            can_id = msg.arbitration_id

            if can_id == 0x000:
                dos_indices.append(index_counter)  
                continue

            if can_id == 0x14A:
                if current_group['14A'] is not None:
                    add_control_obj()
                    current_group = {'14A': None, '17C': None, '1A3': None}
                current_group['14A'] = msg
            elif can_id == 0x17C:
                current_group['17C'] = msg
            elif can_id == 0x1A3:
                current_group['1A3'] = msg

            if all(current_group.values()):
                add_control_obj()
                current_group = {'14A': None, '17C': None, '1A3': None}

    if current_group['14A']:
        add_control_obj()

    return control_obj, dos_indices

def replay_data_without_recovery():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the world
    world = client.get_world()
    # world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    # world.load_map_layer(carla.MapLayer.Buildings)
    # world.load_map_layer(carla.MapLayer.Ground)

    # Configure simulation
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.0125  
    world.apply_settings(settings)

    # Spawn vehicle at the same location
    spawn_points = world.get_map().get_spawn_points()
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle = world.try_spawn_actor(blueprint, spawn_points[11])

    # Convert CAN data to vehicle control
    controls = convert_can_to_control_data('./Logs/merged_output.log')

    #Open log files for writing
    vehicle_location_writer = open('./Logs/vehicle_rep_location.log', 'w')
    vehicle_control_writer = open('./Logs/vehicle_rep_control_log.log', 'w')
    timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')

    # Initialize lists to store data
    vehicle_location = []
    timestamps = []

    # Start simulation
    start_time = timeit.default_timer()
    
    # Replay the control data
    try:
        for control in controls:
            world.tick()

            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

            # Apply control to the vehicle
            vehicle.apply_control(control)

            # Get vehicle location
            location = vehicle.get_location()
            vehicle_location.append(location)

    finally:

        # Write vehicle location and control data to files
        for control in controls:
            vehicle_control_writer.write(str(control) + "\n")
        vehicle_control_writer.close()

        for location in vehicle_location:
            vehicle_location_writer.write(str(location) + "\n")
        vehicle_location_writer.close()

        for ele in timestamps:
            if (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        # Destroy the vehicle and reset settings
        if vehicle:
            vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        plot_gen_vs_rep_paths_from_files("./Logs/vehicle_gen_location.log", "./Logs/vehicle_rep_location.log", "./Graphs/gen_vs_rep_path.png")
        plot_vc_time(controls, timestamps, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")

        print("Replay completed.")

def replay_data_with_recovery_spoof():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the world
    world = client.get_world()
    # world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    # world.load_map_layer(carla.MapLayer.Buildings)
    # world.load_map_layer(carla.MapLayer.Ground)

    # Configure Traffic Manager
    tm = client.get_trafficmanager(8000)  
    tm.set_synchronous_mode(True)  

    # Configure simulation
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.0125  
    world.apply_settings(settings)

    # Spawn vehicle at the same location
    spawn_points = world.get_map().get_spawn_points()
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle = world.try_spawn_actor(blueprint, spawn_points[11])

    recovery_mode = False
    tm_start_index = 5000  # <-- choose when to switch to TM control

    # Convert CAN data to vehicle control
    controls = convert_can_to_control_data('./Logs/merged_output.log')

    #Open log files for writing
    vehicle_location_writer = open('./Logs/vehicle_rep_location.log', 'w')
    vehicle_control_writer = open('./Logs/vehicle_rep_control_log.log', 'w')
    timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')
    recovery_route_locations_reader = open('./Logs/vehicle_gen_route_location.log', 'r')

    # Read the recovery route locations from the file
    recovery_route_locations = []
    pattern = re.compile(r'Location\(x=([-.\d]+), y=([-.\d]+), z=([-.\d]+)\)')

    for line in recovery_route_locations_reader:
        line = line.strip()
        match = pattern.match(line)
        if match:
            x, y, z = map(float, match.groups())
            recovery_route_locations.append(carla.Location(x=x, y=y, z=z))
    recovery_route_locations_reader.close()

    # Initialize lists to store data
    vehicle_location = []
    timestamps = []

    # Start simulation
    start_time = timeit.default_timer()
    
    # Replay the control data
    try:
        for i, control in enumerate(controls):
            world.tick()
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

            if not recovery_mode and i < tm_start_index:
                # Apply manual (replay) control
                vehicle.apply_control(control)
            elif not recovery_mode and i >= tm_start_index:
                # Switch to recovery mode
                vehicle.set_autopilot(True, tm.get_port())
                tm.ignore_lights_percentage(vehicle, 100)
                tm.set_path(vehicle, recovery_route_locations)
                recovery_mode = True

            # Always log location
            location = vehicle.get_location()
            vehicle_location.append(location)

    finally:

        # Write vehicle location and control data to files
        for control in controls:
            vehicle_control_writer.write(str(control) + "\n")
        vehicle_control_writer.close()

        for location in vehicle_location:
            vehicle_location_writer.write(str(location) + "\n")
        vehicle_location_writer.close()

        for ele in timestamps:
            if (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        # Destroy the vehicle and reset settings
        # if vehicle:
        #     vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # plot_gen_vs_rep_paths_from_files("./Logs/vehicle_gen_location.log", "./Logs/vehicle_rep_location.log", "./Graphs/gen_vs_rep_path.png")
        plot_vc_time(controls, timestamps, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")

        vehicle.destroy()

        print("Replay completed.")

def replay_data_with_recovery_dos():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the world
    world = client.get_world()
    # world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    # world.load_map_layer(carla.MapLayer.Buildings)
    # world.load_map_layer(carla.MapLayer.Ground)

    # Configure Traffic Manager
    tm = client.get_trafficmanager(8000)  
    tm.set_synchronous_mode(True)  

    # Configure simulation
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.0125  
    world.apply_settings(settings)

    # Spawn vehicle at the same location
    spawn_points = world.get_map().get_spawn_points()
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle = world.try_spawn_actor(blueprint, spawn_points[11])

    # Convert CAN data to vehicle control
    controls, dos_indices = convert_can_to_control_data_dos('./Logs/merged_output.log')
    
    #Open log files for writing
    vehicle_location_writer = open('./Logs/vehicle_rep_location.log', 'w')
    vehicle_control_writer = open('./Logs/vehicle_rep_control_log.log', 'w')
    timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')
    recovery_route_locations_reader = open('./Logs/vehicle_gen_route_location.log', 'r')

    # Read the recovery route locations from the file
    recovery_route_locations = []
    pattern = re.compile(r'Location\(x=([-.\d]+), y=([-.\d]+), z=([-.\d]+)\)')

    # Read the recovery route locations from the file
    for line in recovery_route_locations_reader:
        line = line.strip()
        match = pattern.match(line)
        if match:
            x, y, z = map(float, match.groups())
            recovery_route_locations.append(carla.Location(x=x, y=y, z=z))
    recovery_route_locations_reader.close()

    # Initialize lists to store data
    vehicle_location = []
    timestamps = []

    recovery_mode = False
    tm_start_index = 3500  # <-- choose when to switch to TM control

    # Start simulation
    start_time = timeit.default_timer()
    
    # Replay the control data
    try:
        for i, control in enumerate(controls):
            world.tick()

            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

            # Add delay for DoS messages
            if i in dos_indices:
                delay = 0
                while(delay < 150000):
                    delay += 1
                # end_time = timeit.default_timer() - start_time
                # print("Delay time: ", end_time-current_time)

            if not recovery_mode and i < tm_start_index:
                vehicle.apply_control(control)
            elif not recovery_mode and i >= tm_start_index:
                vehicle.set_autopilot(True, tm.get_port())
                tm.ignore_lights_percentage(vehicle, 100)
                tm.set_path(vehicle, recovery_route_locations)
                recovery_mode = True

            # Log vehicle location
            location = vehicle.get_location()
            vehicle_location.append(location)

    finally:

        # Write vehicle location and control data to files
        for control in controls:
            vehicle_control_writer.write(str(control) + "\n")
        vehicle_control_writer.close()

        for location in vehicle_location:
            vehicle_location_writer.write(str(location) + "\n")
        vehicle_location_writer.close()

        for ele in timestamps:
            if (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        # Destroy the vehicle and reset settings
        # if vehicle:
        #     vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        plot_gen_vs_rep_paths_from_files("./Logs/vehicle_gen_location.log", "./Logs/vehicle_rep_location.log", "./Graphs/gen_vs_rep_path.png")
        plot_vc_time(controls, timestamps, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")

        vehicle.destroy()

        print("Replay completed.")

def replay_data_with_recovery_dos_2():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the world
    # world = client.get_world()
    world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)

    # Configure simulation
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.0125  
    world.apply_settings(settings)

     # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicle_1 at index 11
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle_2 = world.try_spawn_actor(blueprint, spawn_points[11])

    second_vehicle_transform = spawn_points[11]
    backward_vector = second_vehicle_transform.get_forward_vector() * 25  # 25 meters behind
    first_vehicle_transform = carla.Transform(
        second_vehicle_transform.location + backward_vector,
        second_vehicle_transform.rotation
    )
    vehicle_1 = world.try_spawn_actor(blueprint, first_vehicle_transform)

    # Configure Traffic Manager
    tm = client.get_trafficmanager(8000)  
    tm.set_synchronous_mode(True)  
    tm.set_random_device_seed(9)

    # Enable autopilot and assign traffic manager port
    vehicle_1.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_1, 100)

    vehicle_2.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_2, 100)

    tm.auto_lane_change(vehicle_1, False)
    tm.random_left_lanechange_percentage(vehicle_1, 0)
    tm.random_right_lanechange_percentage(vehicle_1, 0)

    tm.auto_lane_change(vehicle_2, False)
    tm.random_left_lanechange_percentage(vehicle_2, 0)
    tm.random_right_lanechange_percentage(vehicle_2, 0)

    tm.distance_to_leading_vehicle(vehicle_2, 25.0)

    # Convert CAN data to vehicle control
    controls_1, dos_indices = convert_can_to_control_data_dos('./Logs/can_data_logs.log')
    # controls_2 = extract_control_data('./Logs/vehicle_gen_control_log_2.log')

    #Open log files for writing
    vehicle_location_writer_1 = open('./Logs/vehicle_rep_location_1.log', 'w')
    vehicle_location_writer_2 = open('./Logs/vehicle_rep_location_2.log', 'w')
    vehicle_control_writer_1 = open('./Logs/vehicle_rep_control_log_1.log', 'w')
    vehicle_control_writer_2 = open('./Logs/vehicle_rep_control_log_2.log', 'w')
    timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')
    recovery_route_locations_reader = open('./Logs/vehicle_gen_route_location.log', 'r')

    # Read the recovery route locations from the file
    recovery_route_locations = []
    pattern = re.compile(r'Location\(x=([-.\d]+), y=([-.\d]+), z=([-.\d]+)\)')

    for line in recovery_route_locations_reader:
        line = line.strip()
        match = pattern.match(line)
        if match:
            x, y, z = map(float, match.groups())
            recovery_route_locations.append(carla.Location(x=x, y=y, z=z))
    recovery_route_locations_reader.close()

    tm.set_path(vehicle_1, recovery_route_locations)
    tm.set_path(vehicle_2, recovery_route_locations)

    # Initialize lists to store data
    vehicle_location_1 = []
    vehicle_location_2 = []
    vehicle_control_1 = []
    vehicle_control_2 = []
    timestamps = []

    # Start simulation
    start_time = timeit.default_timer()

    tm_end_index = 0
    tm_start_index = 10000

    skipped_control_objects = queue.Queue()

    # clock = pygame.time.Clock()

    # print(dos_indices)
    
    # Replay the control data
    try:
        for i, control_1 in enumerate(controls_1):
            # clock.tick_busy_loop(100)

            world.tick()
            
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

            if i in dos_indices:
                skipped_control_objects.put(control_1)
                pass
            else:
                while not skipped_control_objects.empty():
                    # print(i, skipped_control_objects.qsize())
                    control = skipped_control_objects.get()
                    vehicle_1.apply_control(control)

                if i < tm_end_index:
                    # Phase 1: Pre-Attack — Traffic Manager controls the cars
                    control_1 = vehicle_1.get_control()
                    control_2 = vehicle_2.get_control()
                elif i == tm_end_index:
                    # Phase 2: Attack starts — switch to manual control
                    vehicle_1.set_autopilot(False, tm.get_port())
                    # vehicle_2.set_autopilot(False, tm.get_port())
                    vehicle_1.apply_control(control_1)
                    # vehicle_2.apply_control(control_2)
                elif tm_end_index < i < tm_start_index:
                    # During attack — continue applying manual control
                    vehicle_1.apply_control(control_1)
                    # vehicle_2.apply_control(control_2)
                elif i == tm_start_index:
                    # Phase 3: Recovery — switch autopilot back on
                    vehicle_1.set_autopilot(True, tm.get_port())
                    # vehicle_2.set_autopilot(True, tm.get_port())
                    control_1 = vehicle_1.get_control()
                else:
                    # Post recovery — let TM take over again
                    control_1 = vehicle_1.get_control()

            
            control_2 = vehicle_2.get_control()

            # Log controls
            vehicle_control_1.append(control_1)
            vehicle_control_2.append(control_2)

            # Always log location
            location_1 = vehicle_1.get_location()
            vehicle_location_1.append(location_1)
            location_2 = vehicle_2.get_location()
            vehicle_location_2.append(location_2)

    finally:

        # Write vehicle location and control data to files
        for control in vehicle_control_1:
            vehicle_control_writer_1.write(str(control) + "\n")
        vehicle_control_writer_1.close()

        for control in vehicle_control_2:
            vehicle_control_writer_2.write(str(control) + "\n")
        vehicle_control_writer_2.close()

        for location in vehicle_location_1:
            vehicle_location_writer_1.write(str(location) + "\n")
        vehicle_location_writer_1.close()

        for location in vehicle_location_2:
            vehicle_location_writer_2.write(str(location) + "\n")
        vehicle_location_writer_2.close()

        for ele in timestamps:
            if (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        # Destroy the vehicle and reset settings
        # if vehicle:
        #     vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        plot_gen_vs_rep_paths_from_files("./Logs/vehicle_gen_location_1.log", "./Logs/vehicle_rep_location_1.log", "./Graphs/gen_vs_rep_path.png", 0, 10000)
        # plot_vc_time(vehicle_control_1, timestamps, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")
        # plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs', './Graphs/plot_euclid_diff_gen.png', 8000, 11000)
        plot_euclid_diff_by_index_only(vehicle_location_1, vehicle_location_2,'./Logs', './Graphs/plot_euclid_diff_gen.png', 0, 11000)
        plot_diff(timestamps,"./Graphs/plot_timestamp_diff_rep.png")
        # plot_vc_index(vehicle_control_1, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")

        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Replay completed.")



# def replay_data_with_recovery_dos_2():
#     # Initialize CARLA client
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(10.0)

#     # Load the world
#     world = client.get_world()
#     # world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
#     # world.load_map_layer(carla.MapLayer.Buildings)
#     # world.load_map_layer(carla.MapLayer.Ground)

#     # Configure simulation
#     settings = world.get_settings()
#     settings.synchronous_mode = True 
#     settings.fixed_delta_seconds = 0.0125  
#     world.apply_settings(settings)

#      # Get spawn points
#     spawn_points = world.get_map().get_spawn_points()

#     # Spawn vehicle_1 at index 11
#     blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
#     vehicle_2 = world.try_spawn_actor(blueprint, spawn_points[11])

#     second_vehicle_transform = spawn_points[11]
#     backward_vector = second_vehicle_transform.get_forward_vector() * 25  # 25 meters behind
#     first_vehicle_transform = carla.Transform(
#         second_vehicle_transform.location + backward_vector,
#         second_vehicle_transform.rotation
#     )
#     vehicle_1 = world.try_spawn_actor(blueprint, first_vehicle_transform)

#     # Configure Traffic Manager
#     tm_1 = client.get_trafficmanager(8000)  
#     tm_1.set_synchronous_mode(True)  
#     tm_1.set_random_device_seed(9)

#     tm_2 = client.get_trafficmanager(8001)  
#     tm_2.set_synchronous_mode(False)  
#     tm_2.set_random_device_seed(10)

#     # Enable autopilot and assign traffic manager port
#     vehicle_1.set_autopilot(True, tm_1.get_port())
#     tm_1.ignore_lights_percentage(vehicle_1, 100)

#     vehicle_2.set_autopilot(True, tm_2.get_port())
#     tm_2.ignore_lights_percentage(vehicle_2, 100)

#     tm_1.auto_lane_change(vehicle_1, False)
#     tm_1.random_left_lanechange_percentage(vehicle_1, 0)
#     tm_1.random_right_lanechange_percentage(vehicle_1, 0)

#     tm_2.auto_lane_change(vehicle_2, False)
#     tm_2.random_left_lanechange_percentage(vehicle_2, 0)
#     tm_2.random_right_lanechange_percentage(vehicle_2, 0)

#     tm_2.distance_to_leading_vehicle(vehicle_2, 25.0)

#     # Convert CAN data to vehicle control
#     controls_1, dos_indices = convert_can_to_control_data_dos('./Logs/merged_output.log')
#     # controls_2 = extract_control_data('./Logs/vehicle_gen_control_log_2.log')

#     #Open log files for writing
#     vehicle_location_writer_1 = open('./Logs/vehicle_rep_location_1.log', 'w')
#     vehicle_location_writer_2 = open('./Logs/vehicle_rep_location_2.log', 'w')
#     vehicle_control_writer_1 = open('./Logs/vehicle_rep_control_log_1.log', 'w')
#     vehicle_control_writer_2 = open('./Logs/vehicle_rep_control_log_2.log', 'w')
#     timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')
#     recovery_route_locations_reader = open('./Logs_Benign/vehicle_gen_route_location.log', 'r')

#     # Read the recovery route locations from the file
#     recovery_route_locations = []
#     pattern = re.compile(r'Location\(x=([-.\d]+), y=([-.\d]+), z=([-.\d]+)\)')

#     for line in recovery_route_locations_reader:
#         line = line.strip()
#         match = pattern.match(line)
#         if match:
#             x, y, z = map(float, match.groups())
#             recovery_route_locations.append(carla.Location(x=x, y=y, z=z))
#     recovery_route_locations_reader.close()

#     tm_1.set_path(vehicle_1, recovery_route_locations)
#     tm_2.set_path(vehicle_2, recovery_route_locations)

#     # Initialize lists to store data
#     vehicle_location_1 = []
#     vehicle_location_2 = []
#     vehicle_control_1 = []
#     vehicle_control_2 = []
#     timestamps = []

#     # Start simulation
#     start_time = timeit.default_timer()

#     tm_end_index = 8500
#     tm_start_index = 9800

#     skipped_control_objects = queue.Queue()

#     # print(dos_indices)
    
#     # Replay the control data
#     try:
#         for i, control_1 in enumerate(controls_1):
#             world.tick()
            
#             current_time = timeit.default_timer() - start_time
#             timestamps.append(current_time)

#             if i in dos_indices:
#                 skipped_control_objects.put(control_1)
#                 pass
#             else:
#                 while not skipped_control_objects.empty():
#                     # print(i, skipped_control_objects.qsize())
#                     control = skipped_control_objects.get()
#                     vehicle_1.apply_control(control)

#                 if i < tm_end_index:
#                     # Phase 1: Pre-Attack — Traffic Manager controls the cars
#                     control_1 = vehicle_1.get_control()
#                     control_2 = vehicle_2.get_control()
#                 elif i == tm_end_index:
#                     # Phase 2: Attack starts — switch to manual control
#                     vehicle_1.set_autopilot(False, tm_1.get_port())
#                     # vehicle_2.set_autopilot(False, tm.get_port())
#                     vehicle_1.apply_control(control_1)
#                     # vehicle_2.apply_control(control_2)
#                 elif tm_end_index < i < tm_start_index:
#                     # During attack — continue applying manual control
#                     vehicle_1.apply_control(control_1)
#                     # vehicle_2.apply_control(control_2)
#                 elif i == tm_start_index:
#                     # Phase 3: Recovery — switch autopilot back on
#                     vehicle_1.set_autopilot(True, tm_1.get_port())
#                     # vehicle_2.set_autopilot(True, tm.get_port())
#                     control_1 = vehicle_1.get_control()
#                 else:
#                     # Post recovery — let TM take over again
#                     control_1 = vehicle_1.get_control()

            
#             control_2 = vehicle_2.get_control()

#             # Log controls
#             vehicle_control_1.append(control_1)
#             vehicle_control_2.append(control_2)

#             # Always log location
#             location_1 = vehicle_1.get_location()
#             vehicle_location_1.append(location_1)
#             location_2 = vehicle_2.get_location()
#             vehicle_location_2.append(location_2)

#     finally:

#         # Write vehicle location and control data to files
#         for control in vehicle_control_1:
#             vehicle_control_writer_1.write(str(control) + "\n")
#         vehicle_control_writer_1.close()

#         for control in vehicle_control_2:
#             vehicle_control_writer_2.write(str(control) + "\n")
#         vehicle_control_writer_2.close()

#         for location in vehicle_location_1:
#             vehicle_location_writer_1.write(str(location) + "\n")
#         vehicle_location_writer_1.close()

#         for location in vehicle_location_2:
#             vehicle_location_writer_2.write(str(location) + "\n")
#         vehicle_location_writer_2.close()

#         for ele in timestamps:
#             if (ele < 10):
#                 timestamp_writer.write("0" + str(ele)[:8] + '\n')
#             else:
#                 timestamp_writer.write(str(ele)[:9] + '\n')
#         timestamp_writer.close()

#         # Destroy the vehicle and reset settings
#         # if vehicle:
#         #     vehicle.destroy()
#         settings.synchronous_mode = False
#         world.apply_settings(settings)
#         tm_1.set_synchronous_mode(False)

#         plot_gen_vs_rep_paths_from_files("./Logs/vehicle_gen_location_1.log", "./Logs/vehicle_rep_location_1.log", "./Graphs/gen_vs_rep_path.png", 8500, 10000)
#         # plot_vc_time(vehicle_control_1, timestamps, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")
#         plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs_Benign', './Graphs/plot_euclid_diff_gen.png', 8500, 10000)
#         plot_diff(timestamps,"./Graphs/plot_timestamp_diff_rep.png")
#         plot_vc_index(vehicle_control_1, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")

#         vehicle_1.destroy()
#         vehicle_2.destroy()

#         print("Replay completed.")

def replay_data_with_recovery_online_candump():
    """
        1. Recovery
        2. Online candump
    """

    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the world
    # world = client.get_world()
    world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)

    # Configure simulation
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.0125  
    world.apply_settings(settings)

     # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicle_1 at index 11
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle_2 = world.try_spawn_actor(blueprint, spawn_points[11])

    second_vehicle_transform = spawn_points[11]
    backward_vector = second_vehicle_transform.get_forward_vector() * 25  # 25 meters behind
    first_vehicle_transform = carla.Transform(
        second_vehicle_transform.location + backward_vector,
        second_vehicle_transform.rotation
    )
    vehicle_1 = world.try_spawn_actor(blueprint, first_vehicle_transform)

    # Configure Traffic Manager
    tm = client.get_trafficmanager(8000)  
    tm.set_synchronous_mode(True)  
    tm.set_random_device_seed(9)

    # Enable autopilot and assign traffic manager port
    vehicle_1.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_1, 100)

    vehicle_2.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_2, 100)

    tm.auto_lane_change(vehicle_1, False)
    tm.random_left_lanechange_percentage(vehicle_1, 0)
    tm.random_right_lanechange_percentage(vehicle_1, 0)

    tm.auto_lane_change(vehicle_2, False)
    tm.random_left_lanechange_percentage(vehicle_2, 0)
    tm.random_right_lanechange_percentage(vehicle_2, 0)

    tm.distance_to_leading_vehicle(vehicle_2, 25.0)

    # Convert CAN data to vehicle control
    controls_1 = convert_can_to_control_data('./Logs/can_data_logs.log')
    # controls_2 = extract_control_data('./Logs/vehicle_gen_control_log_2.log')

    #Open log files for writing
    vehicle_location_writer_1 = open('./Logs/vehicle_rep_location_1.log', 'w')
    vehicle_location_writer_2 = open('./Logs/vehicle_rep_location_2.log', 'w')
    vehicle_control_writer_1 = open('./Logs/vehicle_rep_control_log_1.log', 'w')
    vehicle_control_writer_2 = open('./Logs/vehicle_rep_control_log_2.log', 'w')
    timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')
    recovery_route_locations_reader = open('./Logs_Benign/vehicle_gen_route_location.log', 'r')

    # Read the recovery route locations from the file
    recovery_route_locations = []
    pattern = re.compile(r'Location\(x=([-.\d]+), y=([-.\d]+), z=([-.\d]+)\)')

    for line in recovery_route_locations_reader:
        line = line.strip()
        match = pattern.match(line)
        if match:
            x, y, z = map(float, match.groups())
            recovery_route_locations.append(carla.Location(x=x, y=y, z=z))
    recovery_route_locations_reader.close()

    tm.set_path(vehicle_1, recovery_route_locations)
    tm.set_path(vehicle_2, recovery_route_locations)

    # Initialize lists to store data
    vehicle_location_1 = []
    vehicle_location_2 = []
    vehicle_control_1 = []
    vehicle_control_2 = []
    timestamps = []

    can_handler = CAN_Data_Logger()

    # Start simulation
    start_time = timeit.default_timer()

    tm_end_index = 8500
    tm_start_index = 9800

    total_iterations = len(controls_1)

    # Replay the control data
    try:
        for i, control_1 in enumerate(controls_1):
            world.tick()
            
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

            if i < tm_end_index:
                # Phase 1: Pre-Attack — Traffic Manager controls the cars
                control_1 = vehicle_1.get_control()
                control_2 = vehicle_2.get_control()
            elif i == tm_end_index:
                # Phase 2: Attack starts — switch to manual control
                vehicle_1.set_autopilot(False, tm.get_port())
                vehicle_1.apply_control(control_1)
            elif tm_end_index < i < tm_start_index:
                vehicle_1.apply_control(control_1)
            elif i == tm_start_index:
                # Phase 3: Recovery — switch autopilot back on
                vehicle_1.set_autopilot(True, tm.get_port())
                control_1 = vehicle_1.get_control()
            else:
                # Post recovery — let TM take over again
                control_1 = vehicle_1.get_control()

            
            control_2 = vehicle_2.get_control()

            if i >= total_iterations/2:
                current_lights = vehicle_1.get_light_state()
                vehicle_1.set_light_state(carla.VehicleLightState(current_lights | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position))

            # Get vehicle_1 light state
            light_state = vehicle_1.get_light_state()
            left_blinker = int(bool(light_state & carla.VehicleLightState.LeftBlinker))
            right_blinker = int(bool(light_state & carla.VehicleLightState.RightBlinker))
            low_beam = int(bool(light_state & carla.VehicleLightState.LowBeam))
            high_beam = int(bool(light_state & carla.VehicleLightState.HighBeam))
            park_lights = int(bool(light_state & carla.VehicleLightState.Position))
            
            # Get vehicle_1 speed
            velocity = vehicle_1.get_velocity()
            speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

            can_handler.log_data(
                control_1.steer,
                control_1.throttle,
                control_1.brake,
                control_1.gear,
                control_1.manual_gear_shift,
                left_blinker,
                right_blinker,
                low_beam,
                high_beam,
                park_lights,
                control_1.hand_brake,
                speed
            )

            # Log controls
            vehicle_control_1.append(control_1)
            vehicle_control_2.append(control_2)

            # Always log location
            location_1 = vehicle_1.get_location()
            vehicle_location_1.append(location_1)
            location_2 = vehicle_2.get_location()
            vehicle_location_2.append(location_2)

    finally:

        # Write vehicle location and control data to files
        for control in vehicle_control_1:
            vehicle_control_writer_1.write(str(control) + "\n")
        vehicle_control_writer_1.close()

        for control in vehicle_control_2:
            vehicle_control_writer_2.write(str(control) + "\n")
        vehicle_control_writer_2.close()

        for location in vehicle_location_1:
            vehicle_location_writer_1.write(str(location) + "\n")
        vehicle_location_writer_1.close()

        for location in vehicle_location_2:
            vehicle_location_writer_2.write(str(location) + "\n")
        vehicle_location_writer_2.close()

        for ele in timestamps:
            if (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        # Destroy the vehicle and reset settings
        # if vehicle:
        #     vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        plot_gen_vs_rep_paths_from_files("./Logs/vehicle_gen_location_1.log", "./Logs/vehicle_rep_location_1.log", "./Graphs/gen_vs_rep_path.png", 0, 11000)
        # plot_vc_time(vehicle_control_1, timestamps, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")
        # plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs', './Graphs/plot_euclid_diff_gen.png', 8000, 11000)
        plot_euclid_diff_by_index_only(vehicle_location_1, vehicle_location_2,'./Logs', './Graphs/plot_euclid_diff_gen.png', 0, 11000)
        plot_diff(timestamps,"./Graphs/plot_timestamp_diff_rep.png")
        plot_vc_index(vehicle_control_1, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")

        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Replay completed.")

def replay_data_fixed_path():
    """
        1. Offline candump
        2. Single TM
    """

    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the carla_map
    # world = client.load_world('Town05_Opt', reset_settings=True, map_layers=carla.MapLayer.NONE)
    # world.load_map_layer(carla.MapLayer.Buildings)
    # world.load_map_layer(carla.MapLayer.Ground)
    world = client.get_world()

    # Configure Traffic Manager
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(9)

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    # print(spawn_points)

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.0125
    world.apply_settings(settings)

    # Spawn vehicle_1 at index 11
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    vehicle_2 = world.try_spawn_actor(blueprint, spawn_points[11])

    second_vehicle_transform = spawn_points[11]
    backward_vector = second_vehicle_transform.get_forward_vector() * 25  # 25 meters behind
    first_vehicle_transform = carla.Transform(
        second_vehicle_transform.location + backward_vector,
        second_vehicle_transform.rotation
    )
    vehicle_1 = world.try_spawn_actor(blueprint, first_vehicle_transform)

 
    # Enable autopilot and assign traffic manager port
    vehicle_1.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_1, 100)

    vehicle_2.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_2, 100)

    # Pick a destination (ensure it's not the same as the start)
    destination_index = 18
    destination_location = spawn_points[destination_index].location

    # Use GlobalRoutePlanner to generate path
    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, 2.0)
    
    # # Get route
    start_location = vehicle_1.get_location()
    route = grp.trace_route(start_location, destination_location)

    # Extract locations from the route and send to TM
    route_locations = [wp[0].transform.location for wp in route]

    route_locations = []
    pattern = re.compile(r'Location\(x=([-.\d]+), y=([-.\d]+), z=([-.\d]+)\)')
    route_locations_reader = open('./Logs_Benign/vehicle_gen_route_location.log', 'r')

    for line in route_locations_reader:
        line = line.strip()
        match = pattern.match(line)
        if match:
            x, y, z = map(float, match.groups())
            route_locations.append(carla.Location(x=x, y=y, z=z))
    route_locations_reader.close()

    # print(route_locations)
    tm.set_path(vehicle_1, route_locations)
    tm.set_path(vehicle_2, route_locations)

    tm.auto_lane_change(vehicle_1, False)
    tm.random_left_lanechange_percentage(vehicle_1, 0)
    tm.random_right_lanechange_percentage(vehicle_1, 0)

    tm.auto_lane_change(vehicle_2, False)
    tm.random_left_lanechange_percentage(vehicle_2, 0)
    tm.random_right_lanechange_percentage(vehicle_2, 0)

    tm.distance_to_leading_vehicle(vehicle_2, 25.0)

    # Open log files
    vehicle_location_writer_1 = open('./Logs/vehicle_rep_location_1.log', 'w')
    vehicle_location_writer_2 = open('./Logs/vehicle_rep_location_2.log', 'w')
    vehicle_control_writer_1 = open('./Logs/vehicle_rep_control_log_1.log', 'w')
    vehicle_control_writer_2 = open('./Logs/vehicle_rep_control_log_2.log', 'w')
    timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')
    vehicle_route_location = open('./Logs/vehicle_rep_route_location.log', 'w')

    # Data collection
    vehicle_control_obj_1 = []
    vehicle_control_obj_2 = []
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []
    vehicle_velocity = []
    vehicle_light_state = []

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    total_iterations = 11000

    spoof_control = carla.VehicleControl(throttle=1.0, steer=-1.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=1)

    # new_dest_index_array = [217,9]

    # Start simulation
    start_time = timeit.default_timer()
    try:
        for i in range(total_iterations):  
            world.tick()
            
            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

            if i >= 8500 and i <= 9000:
                vehicle_1.apply_control(spoof_control)
                # vehicle_control_obj_1.append(spoof_control)

            # Get current control state
            control_1 = vehicle_1.get_control()
            vehicle_control_obj_1.append(control_1)
            control_2 = vehicle_2.get_control()
            vehicle_control_obj_2.append(control_2)

            if i >= total_iterations/2:
                current_lights = vehicle_1.get_light_state()
                vehicle_1.set_light_state(carla.VehicleLightState(current_lights | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position))

            # Get vehicle_1 light state
            light_state = vehicle_1.get_light_state()
            vehicle_light_state.append(light_state)

            # Get vehicle_1 speed
            velocity = vehicle_1.get_velocity()
            vehicle_velocity.append(velocity)

            # Get vehicle_1 location
            location = vehicle_1.get_location()
            vehicle_location_1.append(location)
            location = vehicle_2.get_location()
            vehicle_location_2.append(location)

    finally:
        print("Starting data logging...")

        # Candump Control Object to CAN
        # for control, light_state, velocity in zip(vehicle_control_obj_1, vehicle_light_state, vehicle_velocity):
        #     steer_data = control.steer
        #     throttle_data = control.throttle
        #     brake_data = control.brake
        #     gear_data = control.gear
        #     manual_gear_shift = control.manual_gear_shift

        #     # Light state breakdown
        #     left_blinker = int(bool(light_state & carla.VehicleLightState.LeftBlinker))
        #     right_blinker = int(bool(light_state & carla.VehicleLightState.RightBlinker))
        #     low_beam = int(bool(light_state & carla.VehicleLightState.LowBeam))
        #     high_beam = int(bool(light_state & carla.VehicleLightState.HighBeam))
        #     park_lights = int(bool(light_state & carla.VehicleLightState.Position))

        #     # Speed (in km/h)
        #     speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        #     can_handler.log_data(
        #         steer_data,
        #         throttle_data,
        #         brake_data,
        #         gear_data,
        #         manual_gear_shift,
        #         left_blinker,
        #         right_blinker,
        #         low_beam,
        #         high_beam,
        #         park_lights,
        #         control.hand_brake,
        #         speed
        #     )

        for control in vehicle_control_obj_1:
            vehicle_control_writer_1.write(str(control) + "\n")
        vehicle_control_writer_1.close()

        for control in vehicle_control_obj_2:
            vehicle_control_writer_2.write(str(control) + "\n")
        vehicle_control_writer_2.close()

        for ele in timestamps:
            if ele < 10:
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

        for location in route_locations:
            vehicle_route_location.write(str(location) + "\n")
        vehicle_route_location.close()

        plot_vc_index(
            vehicle_control_obj_1,
            "./Graphs/plot_throttle_time_rep.png",
            "./Graphs/plot_steer_time_rep.png",
            "./Graphs/plot_brake_time_rep.png"
        )

        plot_gen_vs_rep_paths_from_files(
            "./Logs/vehicle_gen_location_1.log",
            "./Logs/vehicle_rep_location_1.log",
            "./Graphs/gen_vs_rep_path.png",
            0,
            11000
        )
        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_rep.png")
        plot_euclid_diff_by_index_only(vehicle_location_1, vehicle_location_2, './Logs', './Graphs/plot_euclid_diff.png', 0, 11000)

        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")

if __name__ == '__main__':
    print("Starting replay...")
    replay_data_with_recovery_dos_2()
