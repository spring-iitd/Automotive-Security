import glob
import os
import re
import sys
import timeit

import pygame
import carla
import can
import cantools
from plot_graphs import plot_diff, plot_euclid_diff, plot_euclid_diff_single_function, plot_gen_path_only_from_list, plot_vc_index, plot_vc_time

"""
    This script generates control data for a vehicle and converts them to CAN data in the CARLA simulator and logs it to files.
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
        headlight_data = 1 if (low_beam | high_beam == 1) else 0
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

    def log_data(self, steer_data, throttle_data, brake_data, gear_data, manual_gear_shift):
        self.log_steer_data(steer_data)
        self.log_throttle_brake_data(throttle_data, brake_data)
        self.log_gear_data(gear_data, manual_gear_shift)

def modify_vehicle_physics(vehicle):
    physics_control = vehicle.get_physics_control()
    physics_control.use_sweeping_control = True
    vehicle.apply_physics_control(physics_control)

def generate_data_single_car():
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
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    source = spawn_points[11]
    source.location.y -= 25
    vehicle = world.try_spawn_actor(blueprint, source)
    world.wait_for_tick()

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.002  
    world.apply_settings(settings)

    agent = BasicAgent(vehicle, target_speed=100)
    # agent = BehaviorAgent(vehicle, behavior='normal')
    agent.ignore_traffic_lights(True)  # Enable traffic light compliance
    agent.ignore_stop_signs(True)      # Enable stop sign compliance
    agent.ignore_vehicles(True) 

    new_index = 20
    destination = spawn_points[new_index].location
    agent.set_destination(destination)

    # Open log files
    vehicle_location_writer = open('./Logs/vehicle_gen_location.log', 'w')
    vehicle_control_writer = open('./Logs/vehicle_gen_control_log.log', 'w')
    timestamp_writer = open('./Logs/vehicle_gen_timestamp_log.log', 'w')

    # Initialize lists to store data
    vehicle_control_obj = []
    timestamps = []
    vehicle_location = []

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    sim_seconds = 20

    # Start simulation
    start_time = timeit.default_timer()
    try:
        while True:  
            world.tick()
            
            # Get current time
            current_time = timeit.default_timer() - start_time

            if current_time >= sim_seconds:
                break

            timestamps.append(current_time)

            # Get current control state
            control = agent.run_step()
            vehicle.apply_control(control)
            # can_handler.log_data(control.steer, control.throttle, control.brake, control.gear, control.manual_gear_shift)
            vehicle_control_obj.append(control)

            # Get vehicle location
            location = vehicle.get_location()
            vehicle_location.append(location)
   
    finally:
        
        print("Terminating simulation...")
        print()

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

        plot_vc_index(vehicle_control_obj, "./Graphs/plot_throttle_time_gen.png", "./Graphs/plot_steer_time_gen.png", "./Graphs/plot_brake_time_gen.png")
        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_gen.png")
        # plot_gen_path_only_from_list(vehicle_location, "./Graphs/plot_gen_path.png", 0, total_iterations)

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        # tm.set_synchronous_mode(False)
        
        vehicle.destroy()

        print("Generation completed and data logged.")

def generate_data_two_car():
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
    vehicle_location_writer_1 = open('./Logs/gen_coord_1.log', 'w')
    vehicle_control_writer_1 = open('./Logs/gen_control_obj_1.log', 'w')

    vehicle_location_writer_2 = open('./Logs/gen_coord_2.log', 'w')
    vehicle_control_writer_2 = open('./Logs/gen_control_obj_2.log', 'w')

    timestamp_writer = open('./Logs/gen_timestamps.log', 'w')

    # Initialize lists to store data
    vehicle_control_obj_1 = []
    vehicle_control_obj_2 = []
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    clock = pygame.time.Clock()

    sim_seconds = 65

    # Start simulation
    start_time = timeit.default_timer()
    # current_time = timeit.default_timer() - start_time
    try:
        while True:  
            # clock.tick_busy_loop(400)
            # Get current time
            current_time = timeit.default_timer() - start_time

            if current_time >= sim_seconds:
                break
            
            # print(str(current_time))
            timestamps.append(current_time)

            world.tick()

            # Get current control state
            control_1 = agent_1.run_step()
            control_2 = agent_2.run_step()

            vehicle_1.apply_control(control_1)
            vehicle_2.apply_control(control_2)
            # can_handler.log_data(control.steer, control.throttle, control.brake, control.gear, control.manual_gear_shift)
            vehicle_control_obj_1.append(control_1)
            vehicle_control_obj_2.append(control_2)

            # Get vehicle location
            location_1 = vehicle_1.get_location()
            location_2 = vehicle_2.get_location()
            vehicle_location_1.append(location_1)
            vehicle_location_2.append(location_2)

            delay_time = current_time + 0.004
            current_time = timeit.default_timer() - start_time
            while current_time < delay_time:
                current_time = timeit.default_timer() - start_time

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

        # plot_vc_time(vehicle_control_obj_1, timestamps, "./Graphs/plot_throttle_time_gen.png", "./Graphs/plot_steer_time_gen.png", "./Graphs/plot_brake_time_gen.png")
        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_gen.png")
        plot_euclid_diff(vehicle_location_1, vehicle_location_2, timestamps, "./Graphs/plot_euclid_diff_benign.png")
        # plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs_bkp','./Graphs/plot_euclid_diff_gen.png')

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        # tm.set_synchronous_mode(False)
        
        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")


if __name__ == '__main__':
    print("Starting generation...")
    generate_data_two_car()
