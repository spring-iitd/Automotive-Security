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
from plot_graphs import plot_diff, plot_euclid_diff, plot_euclid_diff_by_index_only, plot_euclid_diff_single_function, plot_gen_vs_rep_paths_from_files, plot_vc_index, plot_vc_time

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

def convert_can_to_control_data(file_path):
    control_obj = []
    can_logger = CAN_Data_Logger()
    
    current_group = {'14A': None, '17C': None, '1A3': None}

    def add_control_obj():
        steer_msg = current_group['14A']
        throttle_msg = current_group['17C']
        gear_msg = current_group['1A3']
        
        throttle, steer, brake, _, _, _, gear = can_logger.parse_logs(steer_msg, throttle_msg, gear_msg)
        
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
                current_group['14A'] = msg

            elif can_id == 0x17C:
                current_group['17C'] = msg

            elif can_id == 0x1A3:
                current_group['1A3'] = msg

            # If all three messages are present, process
            if all(current_group.values()):
                add_control_obj()
                current_group = {'14A': None, '17C': None, '1A3': None}

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

def replay_data():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the map
    # world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    # world.load_map_layer(carla.MapLayer.Buildings)
    # world.load_map_layer(carla.MapLayer.Ground)
    world = client.get_world()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.009  
    world.apply_settings(settings)

    # Spawn vehicle
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    source = spawn_points[20]
    # source.location.y -= 25
    vehicle = world.try_spawn_actor(blueprint, source)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(9)

    controls = convert_can_to_control_data('./Logs/can_data_logs.log')

    #Open log files for writing
    vehicle_location_writer = open('./Logs/vehicle_rep_location.log', 'w')
    vehicle_control_writer = open('./Logs/vehicle_rep_control_log.log', 'w')
    timestamp_writer = open('./Logs/vehicle_rep_timestamp_log.log', 'w')

    # Initialize lists to store data
    vehicle_location = []
    timestamps = []
    vehicle_control = []

    total_iterations = len(controls)

    # Start simulation
    start_time = timeit.default_timer()

    # Replay the control data
    try:
        for control in controls:
            world.tick()

            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)
            vehicle_control.append(control)
            # Apply control to the vehicle
            vehicle.apply_control(control)

            # Get vehicle location
            location = vehicle.get_location()
            vehicle_location.append(location)

    finally:

        # Write vehicle location and control data to files
        for control in vehicle_control:
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

        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_rep.png")
        plot_vc_index(
            vehicle_control,
            "./Graphs/plot_throttle_time_rep.png",
            "./Graphs/plot_steer_time_rep.png",
            "./Graphs/plot_brake_time_rep.png"
        )

        plot_gen_vs_rep_paths_from_files(
            "./Logs/vehicle_gen_location.log",
            "./Logs/vehicle_rep_location.log",
            "./Graphs/gen_vs_rep_path.png",
            0,
            total_iterations
        )

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        
        vehicle.destroy()

       

        print("Replay completed.")

def replay_data_two_car():
    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the map
    # world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    # world.load_map_layer(carla.MapLayer.Buildings)
    # world.load_map_layer(carla.MapLayer.Ground)
    world = client.get_world()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicle
    # car_model_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
    # car_model = car_model_list[2]
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    source = spawn_points[11]
    vehicle_1 = world.try_spawn_actor(blueprint, source)
    world.wait_for_tick()
    source.location.y -= 25
    vehicle_2 = world.try_spawn_actor(blueprint, source)
    world.wait_for_tick()

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.006
    world.apply_settings(settings)

    # Open log files
    vehicle_location_writer_1 = open('./Logs/rep_coord_1.log', 'w')
    vehicle_control_writer_1 = open('./Logs/rep_control_obj_1.log', 'w')

    vehicle_location_writer_2 = open('./Logs/rep_coord_2.log', 'w')
    vehicle_control_writer_2 = open('./Logs/rep_control_obj_2.log', 'w')

    timestamp_writer = open('./Logs/rep_timestamps.log', 'w')

    timestamp_reader = open('./Logs/gen_timestamps.log', 'r')

    # Initialize lists to store data
    vehicle_control_obj_1 = extract_control_data('./Logs/gen_control_obj_1.log')
    vehicle_control_obj_2 = extract_control_data('./Logs/gen_control_obj_2.log')
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []
    gen_timestamps = []

    # Read timestamps from the generated log file
    gen_timestamps = [float(line.strip()) for line in timestamp_reader if line.strip()]

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    clock = pygame.time.Clock()

    # Start simulation
    start_time = timeit.default_timer()
    try:
        for i in range(len(vehicle_control_obj_1)):  
            # clock.tick_busy_loop(400)
            
            # Get current time
            current_time = timeit.default_timer() - start_time

            while current_time < gen_timestamps[i]:
                current_time = timeit.default_timer() - start_time

            timestamps.append(current_time)

            world.tick()
            
            vehicle_1.apply_control(vehicle_control_obj_1[i])
            vehicle_2.apply_control(vehicle_control_obj_2[i])
            # can_handler.log_data(control.steer, control.throttle, control.brake, control.gear, control.manual_gear_shift)
            vehicle_control_obj_1.append(vehicle_control_obj_1[i])
            vehicle_control_obj_2.append(vehicle_control_obj_2[i])

            # Get vehicle location
            location_1 = vehicle_1.get_location()
            location_2 = vehicle_2.get_location()
            vehicle_location_1.append(location_1)
            vehicle_location_2.append(location_2)

            # delay_time = current_time + 0.0035
            # current_time = timeit.default_timer() - start_time
            # while current_time < delay_time:
            #     current_time = timeit.default_timer() - start_time
   
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

        # plot_vc_time(vehicle_control_obj_1, timestamps, "./Graphs/plot_throttle_time_rep.png", "./Graphs/plot_steer_time_rep.png", "./Graphs/plot_brake_time_rep.png")
        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_rep.png")
        # plot_euclid_diff(vehicle_location_1, vehicle_location_2, timestamps, "./Graphs/plot_euclid_diff_benign.png")
        plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs','./Graphs/plot_euclid_diff_rep.png')
        plot_gen_vs_rep_paths_from_files(
            "./Logs/gen_coord_1.log",
            "./Logs/rep_coord_1.log",
            "./Graphs/gen_vs_rep_path.png",
            0,
            70000
        )

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        # tm.set_synchronous_mode(False)
        
        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Replay completed and data logged.")

if __name__ == "__main__":
    replay_data_two_car()
