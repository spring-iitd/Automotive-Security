import glob
import os
import random
import re
import sys
import timeit

import pygame
import carla
import can
import cantools
from plot_graphs import plot_diff, plot_euclid_diff, plot_euclid_diff_by_index_only, plot_euclid_diff_single_function, plot_gen_path_only_from_list, plot_gen_vs_rep_paths_from_files, plot_vc_index, plot_vc_time

"""
    This script generates control data for a vehicle_1 and converts them to CAN data in the CARLA simulator and logs it to files.
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


def generate_data():
    """
    1. Offline candump
    2. Single TM
    3. Single Vehicle
    4. Random selected path
    """

    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the carla_map
    world = client.load_world('Town01_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)

    # Configure Traffic Manager
    tm = client.get_trafficmanager(8000)  
    tm.set_synchronous_mode(True)  

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.0125  
    world.apply_settings(settings)

    # Spawn vehicle_1
    blueprint = world.get_blueprint_library().filter('vehicle_1.tesla.model3')[0]
    vehicle_1 = world.try_spawn_actor(blueprint, spawn_points[11])

    # Enable autopilot and ignore traffic lights
    vehicle_1.set_autopilot(True, tm.get_port())  
    tm.update_vehicle_lights(vehicle_1, True)  # Set vehicle_1 lights to position
    tm.ignore_lights_percentage(vehicle_1, 100)  # Vehicle ignores all traffic lights

    # Open log files
    vehicle_location_writer = open('./Logs/vehicle_gen_location.log', 'w')
    vehicle_control_writer = open('./Logs/vehicle_gen_control_log.log', 'w')
    timestamp_writer = open('./Logs/vehicle_gen_timestamp_log.log', 'w')

    # Initialize lists to store data
    vehicle_control_obj = []
    vehicle_light_state = []
    timestamps = []
    vehicle_location = []
    vehicle_velocity = []

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    total_iterations = 25000

    night_mode_set = False 

    # Start simulation
    start_time = timeit.default_timer()
    try:
        for i in range(total_iterations):  
            world.tick()
            
            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

            # Get current control state
            control = vehicle_1.get_control()
            # can_handler.log_data(control.steer, control.throttle, control.brake, control.gear, control.manual_gear_shift)
            vehicle_control_obj.append(control)

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
            vehicle_location.append(location)
   
    finally:
        
        # Candump Control Object to CAN
        for control, light_state, velocity in zip(vehicle_control_obj, vehicle_light_state, vehicle_velocity):
            steer_data = control.steer
            throttle_data = control.throttle
            brake_data = control.brake
            gear_data = control.gear
            manual_gear_shift = control.manual_gear_shift

            # Light state breakdown
            left_blinker = int(bool(light_state & carla.VehicleLightState.LeftBlinker))
            right_blinker = int(bool(light_state & carla.VehicleLightState.RightBlinker))
            low_beam = int(bool(light_state & carla.VehicleLightState.LowBeam))
            high_beam = int(bool(light_state & carla.VehicleLightState.HighBeam))
            park_lights = int(bool(light_state & carla.VehicleLightState.Position))

            # Speed (in km/h)
            speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

            can_handler.log_data(
                steer_data,
                throttle_data,
                brake_data,
                gear_data,
                manual_gear_shift,
                left_blinker,
                right_blinker,
                low_beam,
                high_beam,
                park_lights,
                control.hand_brake,
                speed
            )


        for control in vehicle_control_obj:
            vehicle_control_writer.write(str(control) + "\n")
        vehicle_control_writer.close()
        
        for ele in timestamps:
            if (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
        timestamp_writer.close()

        for location in vehicle_location:
            vehicle_location_writer.write(str(location) + "\n")
        vehicle_location_writer.close()

        plot_vc_time(vehicle_control_obj, timestamps, "./Graphs/plot_throttle_time_gen.png", "./Graphs/plot_steer_time_gen.png", "./Graphs/plot_brake_time_gen.png")

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        vehicle_1.destroy()

        print("Generation completed and data logged.")

def generate_data_fixed_path():
    """
        1. Offline candump
        2. Single TM
        3. Two Vehicles
        4. Fixed path
    """

    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the carla_map
    world = client.load_world('Town05_Opt', reset_settings=True, map_layers=carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)
    # world = client.get_world()

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
    route_locations_reader = open('./Logs_Benign_TM_bkp/vehicle_gen_route_location.log', 'r')

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
    vehicle_location_writer_1 = open('./Logs/vehicle_gen_location_1.log', 'w')
    vehicle_location_writer_2 = open('./Logs/vehicle_gen_location_2.log', 'w')
    vehicle_control_writer_1 = open('./Logs/vehicle_gen_control_log_1.log', 'w')
    vehicle_control_writer_2 = open('./Logs/vehicle_gen_control_log_2.log', 'w')
    timestamp_writer = open('./Logs/vehicle_gen_timestamp_log.log', 'w')
    vehicle_route_location = open('./Logs/vehicle_gen_route_location.log', 'w')

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

    # clock = pygame.time.Clock()

    total_iterations = 5000

    # spoof_control = carla.VehicleControl(throttle=1.0, steer=-1.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=1)

    # new_dest_index_array = [217,9]

    # Start simulation
    start_time = timeit.default_timer()
    try:
        for i in range(total_iterations):  
            # clock.tick_busy_loop(100)

            world.tick()
            
            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

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
        for control, light_state, velocity in zip(vehicle_control_obj_1, vehicle_light_state, vehicle_velocity):
            steer_data = control.steer
            throttle_data = control.throttle
            brake_data = control.brake
            gear_data = control.gear
            manual_gear_shift = control.manual_gear_shift

            # Light state breakdown
            left_blinker = int(bool(light_state & carla.VehicleLightState.LeftBlinker))
            right_blinker = int(bool(light_state & carla.VehicleLightState.RightBlinker))
            low_beam = int(bool(light_state & carla.VehicleLightState.LowBeam))
            high_beam = int(bool(light_state & carla.VehicleLightState.HighBeam))
            park_lights = int(bool(light_state & carla.VehicleLightState.Position))

            # Speed (in km/h)
            speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

            can_handler.log_data(
                steer_data,
                throttle_data,
                brake_data,
                gear_data,
                manual_gear_shift,
                left_blinker,
                right_blinker,
                low_beam,
                high_beam,
                park_lights,
                control.hand_brake,
                speed
            )

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

        # plot_vc_index(
        #     vehicle_control_obj_1,
        #     "./Graphs/plot_throttle_time_gen.png",
        #     "./Graphs/plot_steer_time_gen.png",
        #     "./Graphs/plot_brake_time_gen.png"
        # )

        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_gen.png")
        plot_euclid_diff(vehicle_location_1, vehicle_location_2, timestamps, "./Graphs/plot_euclid_diff_benign.png")
        # plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs_Benign', './Graphs/plot_euclid_diff_gen.png')

        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")

def generate_data_online_candump():
    """
        1. Online candump
        2. Single TM
        3. Fixed path
        4. Two vehicles
    """

    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the carla_map
    world = client.load_world('Town05_Opt', reset_settings=True, map_layers=carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)
    # world = client.get_world()

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
    vehicle_location_writer_1 = open('./Logs/vehicle_gen_location_1.log', 'w')
    vehicle_location_writer_2 = open('./Logs/vehicle_gen_location_2.log', 'w')
    vehicle_control_writer_1 = open('./Logs/vehicle_gen_control_log_1.log', 'w')
    vehicle_control_writer_2 = open('./Logs/vehicle_gen_control_log_2.log', 'w')
    timestamp_writer = open('./Logs/vehicle_gen_timestamp_log.log', 'w')
    vehicle_route_location = open('./Logs/vehicle_gen_route_location.log', 'w')

    # Data collection
    vehicle_control_obj_1 = []
    vehicle_control_obj_2 = []
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    total_iterations = 11000

    # spoof_control = carla.VehicleControl(throttle=1.0, steer=1.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=1)

    # new_dest_index_array = [217,9]

    # Start simulation
    start_time = timeit.default_timer()
    try:
        for i in range(total_iterations):  
            world.tick()
            
            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

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

            # Get vehicle_1 location
            location = vehicle_1.get_location()
            vehicle_location_1.append(location)
            location = vehicle_2.get_location()
            vehicle_location_2.append(location)

    finally:
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
            "./Graphs/plot_throttle_time_gen.png",
            "./Graphs/plot_steer_time_gen.png",
            "./Graphs/plot_brake_time_gen.png"
        )

        # plot_gen_vs_rep_paths_from_files("./Logs/vehicle_gen_location.log", "./Logs/vehicle_rep_location.log", "./Graphs/gen_vs_rep_path.png")
        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_gen.png")
        plot_euclid_diff(vehicle_location_1, vehicle_location_2, timestamps, "./Graphs/plot_euclid_diff_benign.png")
        plot_gen_path_only_from_list(vehicle_location_1, "./Graphs/plot_gen_path.png", 0, total_iterations)
        # plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs_Benign', './Graphs/plot_euclid_diff_gen.png')

        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")

def generate_data_offline_candump():
    """
    1. Offline candump
    2. Single TM
    3. Two Vehicles
    4. Random selected path
    """

    # Initialize CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the carla_map
    world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
    world.load_map_layer(carla.MapLayer.Buildings)
    world.load_map_layer(carla.MapLayer.Ground)

    spawn_points = world.get_map().get_spawn_points()

    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    source = spawn_points[11]
    vehicle_1 = world.try_spawn_actor(blueprint, source)
    world.wait_for_tick()
    source.location.y -= 25
    vehicle_2 = world.try_spawn_actor(blueprint, source)
    world.wait_for_tick()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(9)

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.0125
    world.apply_settings(settings)

    # Enable autopilot and ignore traffic lights
    vehicle_1.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_1, 100)  # Vehicle ignores all traffic lights
    tm.update_vehicle_lights(vehicle_1, True)  # Set vehicle_1 lights to position
    
    vehicle_2.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle_2, 100)  # Vehicle ignores all traffic lights
    tm.update_vehicle_lights(vehicle_2, True)  # Set vehicle_1 lights to position

    tm.auto_lane_change(vehicle_1, False)
    tm.random_left_lanechange_percentage(vehicle_1, 0)
    tm.random_right_lanechange_percentage(vehicle_1, 0)

    tm.auto_lane_change(vehicle_2, False)
    tm.random_left_lanechange_percentage(vehicle_2, 0)
    tm.random_right_lanechange_percentage(vehicle_2, 0)

    tm.distance_to_leading_vehicle(vehicle_2, 25.0)

    # Open log files
    vehicle_location_writer_1 = open('./Logs/vehicle_gen_location_1.log', 'w')
    vehicle_location_writer_2 = open('./Logs/vehicle_gen_location_2.log', 'w')
    vehicle_control_writer_1 = open('./Logs/vehicle_gen_control_log_1.log', 'w')
    vehicle_control_writer_2 = open('./Logs/vehicle_gen_control_log_2.log', 'w')
    timestamp_writer = open('./Logs/vehicle_gen_timestamp_log.log', 'w')
    vehicle_route_location = open('./Logs/vehicle_gen_route_location.log', 'w')

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

    clock = pygame.time.Clock()

    total_iterations = 5000

    # Start simulation
    start_time = timeit.default_timer()
    try:
        for i in range(total_iterations):  
            # clock.tick_busy_loop(100) 

            world.tick()
            
            # Get current time
            current_time = timeit.default_timer() - start_time
            timestamps.append(current_time)

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
        for control, light_state, velocity in zip(vehicle_control_obj_1, vehicle_light_state, vehicle_velocity):
            steer_data = control.steer
            throttle_data = control.throttle
            brake_data = control.brake
            gear_data = control.gear
            manual_gear_shift = control.manual_gear_shift

            # Light state breakdown
            left_blinker = int(bool(light_state & carla.VehicleLightState.LeftBlinker))
            right_blinker = int(bool(light_state & carla.VehicleLightState.RightBlinker))
            low_beam = int(bool(light_state & carla.VehicleLightState.LowBeam))
            high_beam = int(bool(light_state & carla.VehicleLightState.HighBeam))
            park_lights = int(bool(light_state & carla.VehicleLightState.Position))

            # Speed (in km/h)
            speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

            can_handler.log_data(
                steer_data,
                throttle_data,
                brake_data,
                gear_data,
                manual_gear_shift,
                left_blinker,
                right_blinker,
                low_beam,
                high_beam,
                park_lights,
                control.hand_brake,
                speed
            )

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

        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_gen.png")
        plot_euclid_diff(vehicle_location_1, vehicle_location_2, timestamps, "./Graphs/plot_euclid_diff_benign.png")
        plot_gen_path_only_from_list(vehicle_location_1, "./Graphs/plot_gen_path.png", 0, total_iterations)

        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")


if __name__ == '__main__':
    print("Starting generation...")
    # generate_data_offline_candump()
    generate_data_fixed_path()
    # generate_data_online_candump()
