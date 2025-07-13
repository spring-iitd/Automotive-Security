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
    This script generates control data for a vehicle and converts them to CAN data in the CARLA simulator and logs it to files.
"""

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

# ==============================================================================
# -- CAN Data Logger Class ---------------------------------------------------------
# This class handles the logging of various vehicle data to CAN messages.
# It uses the cantools library to encode messages based on a DBC file and sends them
# over a virtual CAN bus using the python-can library.
# ==============================================================================
class CAN_Data_Logger_Realtime(object):
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

    # This method logs the steering data by encoding it into a CAN message
    def log_steer_data(self, steer_data):
        encoded_steer_data = self.steer_message.encode({'STEER_ANGLE': steer_data})
        message = can.Message(arbitration_id=self.steer_message.frame_id, data=encoded_steer_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    # This method logs the throttle and brake data by encoding it into a CAN message
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

    # This method logs the gear data by encoding it into a CAN message   
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

    # This method logs the blinker data by encoding it into a CAN message
    def log_blinker_data(self, left_blinker, right_blinker):
        encoded_blinker_data = self.blinker_message.encode({'DRIVERS_DOOR_OPEN': 0, 'MAIN_ON': 0,'RIGHT_BLINKER': right_blinker,'LEFT_BLINKER': left_blinker, 'CMBS_STATES': 0})
        message = can.Message(arbitration_id=self.blinker_message.frame_id, data=encoded_blinker_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    # This method logs the headlight data by encoding it into a CAN message
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

    # This method logs the beam data by encoding it into a CAN message
    def log_beam_data(self, low_beam, high_beam, park_lights):
        encoded_beam_data = self.beam_message.encode({'WIPERS': 0, 'LOW_BEAMS': low_beam, 'HIGH_BEAMS': high_beam, 'PARK_LIGHTS': park_lights})
        message = can.Message(arbitration_id=self.beam_message.frame_id, data=encoded_beam_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    # This method logs the handbrake data by encoding it into a CAN message
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

    # This method logs the speed data by encoding it into a CAN message
    def log_speed_data(self, speed):
        encoded_speed_data = self.speed_message.encode({'CAR_SPEED': speed})
        message = can.Message(arbitration_id=self.speed_message.frame_id, data=encoded_speed_data)
        # line = str(message)
        # match = re.search(self.pattern, line)
        # id_value, data_value = match.group(1), match.group(2).replace(" ", "")
        # cmd = "cansend vcan0 " + str(id_value) + "#" + str(data_value)
        # os.system(cmd)
        self.bus.send(message)

    # This method injects a Denial of Service (DoS) message into the CAN bus
    def inject_dos(self):
        # To do DoS with different arbitration IDs, you can modify the arbitration_id and data fields of the message. 
        message = can.Message(arbitration_id=0x00000000, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)

    def start_frame(self):
        # To do DoS with different arbitration IDs, you can modify the arbitration_id and data fields of the message. 
        message = can.Message(arbitration_id=0x1FFFFFFF, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)

    # These methods log dummy data to simulate various CAN messages
    def log_dummy_data_1(self):
        # To send different dummy data, you can modify the arbitration_id and data fields of the message.
        data=[0x00] * 8
        data[0] = random.randint(0, 255) # Random data for the first byte
        message = can.Message(arbitration_id=0x00000170, data=data, is_extended_id=True)
        self.bus.send(message)

    # These methods log dummy data to simulate various CAN messages
    def log_dummy_data_2(self):
        data=[0x00] * 8
        data[1] = random.randint(0, 255) # Random data for the second byte
        message = can.Message(arbitration_id=0x00000202, data=data, is_extended_id=True)
        self.bus.send(message)
    
    def log_dummy_data_3(self):
        data=[0x00] * 8
        data[2] = random.randint(0, 255)
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

    # This method logs all the data at once, checking the index and time difference to determine when to log each message
    def log_realtime_data(self, index, time_diff_tick, steer_data, throttle_data, brake_data, gear_data,
             manual_gear_shift, left_blinker, right_blinker, low_beam, high_beam,
             park_lights, handbrake, speed_data, jitter_array):

        """
        Logs vehicle data as CAN messages based on index and time intervals.

        Each message specification defines:
        - A unique message name
        - A lambda function to call the appropriate logging method with the necessary data
        - A periodicity value (in seconds), indicating how often the message should be logged:
            - 0.0: log every simulation tick
            - > 0.0: log at fixed intervals (e.g., 0.5 logs every 0.5 seconds)

        Additional features:
        - `jitter_array`: introduces controlled randomness to logging intervals
        - `add_delay`: inserts a short delay after sending each message to simulate real-world timing

        The `message_specs` list holds all message definitions as tuples:
            (message_name, log_function, periodicity)
        """

        message_specs = [
            ("steer", lambda: self.log_steer_data(steer_data), 0.0),           # every tick
            ("throttle_brake", lambda: self.log_throttle_brake_data(throttle_data, brake_data), 0.0),
            ("speed", lambda: self.log_speed_data(speed_data), 0.0),
            ("gear", lambda: self.log_gear_data(gear_data, manual_gear_shift), 0.5),
            ("blinker", lambda: self.log_blinker_data(left_blinker, right_blinker), 1.0),
            ("headlight", lambda: self.log_headlight_data(low_beam, high_beam), 1.0),
            ("beam", lambda: self.log_beam_data(low_beam, high_beam, park_lights), 1.0),
            ("handbrake", lambda: self.log_handbrake_data(handbrake), 1.0),
            ("dummy_1", lambda: self.log_dummy_data_1(), 0.0),  # every tick
            ("dummy_2", lambda: self.log_dummy_data_2(), 0.25),
            ("dummy_3", lambda: self.log_dummy_data_3(), 0.5),
            # ("dummy_4", lambda: self.log_dummy_data_4(), 0.75),
            # ("dummy_5", lambda: self.log_dummy_data_5(), 1.0),
            # ("dummy_6", lambda: self.log_dummy_data_6(), 2.0),
            # Example: Add a new message here with 500ms periodicity:
            # ("my_new_msg", lambda: self.log_new_data(data), 0.5),
        ]

        num_msgs = len(message_specs)

        # Determines whether the current index qualifies for logging based on message periodicity.
        # - If periodicity is 0.0 → log on every tick.
        # - Otherwise → log only when the current index aligns with the defined logging interval,
        #   i.e., when the index is a multiple of the number of ticks per period.
        def is_eligible(period):
            if period == 0.0:
                return True
            ticks_per_period = int(period / time_diff_tick) if time_diff_tick > 0 else 1
            return index % ticks_per_period == 0

        # Calculate the chunk of jitter values for the current index
        jitter_chunk = jitter_array[index * num_msgs : (index + 1) * num_msgs]

        eligible_actions = []

        # Iterate through message specifications and check if they are eligible for logging
        # based on the current index and periodicity.
        for i, (name, log_fn, period) in enumerate(message_specs):
            if is_eligible(period):
                jitter = jitter_chunk[i]
                eligible_actions.append((jitter, log_fn))

        # Sort eligible actions by their jitter value to introduce controlled randomness in logging
        sorted_actions = sorted(eligible_actions, key=lambda x: x[0])
        for _, action in sorted_actions:
            action()
            add_delay(0.000200) #100 microseconds

    # This method is called when the object is deleted, shutting down the CAN bus
    def __del__(self):
        self.bus.shutdown()
        pass


class CAN_Data_Logger_Offline(object):
    def __init__(self, world):
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
        self.world = world
        self.bus = can.Bus(channel='vcan0', interface='socketcan')


    def log_steer_data(self, steer_data):
        encoded_steer_data = self.steer_message.encode({'STEER_ANGLE': steer_data})
        message = can.Message(arbitration_id=self.steer_message.frame_id, data=encoded_steer_data)
        self.bus.send(message)

    def log_throttle_brake_data(self, throttle_data, brake_data):
        encoded_throttle_brake_data = self.throttle_brake_message.encode({'PEDAL_GAS': throttle_data, 'ENGINE_RPM': 0, 'GAS_PRESSED': 0, 
                                                   'ACC_STATUS': 0, 'BOH_17C': 0, 'BRAKE_SWITCH': 0, 
                                                   'BOH2_17C': 0, 'BRAKE_PRESSED': brake_data})
        message = can.Message(arbitration_id=self.throttle_brake_message.frame_id, data=encoded_throttle_brake_data)
        self.bus.send(message)
       
    def log_gear_data(self, gear_data, manual_gear_shift):
        manual_gear_shift = 1 if (manual_gear_shift == True) else 0   
        if manual_gear_shift:    #1 for manual and 0 for auto transmission
            encoded_gear_data = self.gear_message.encode({'GEAR_SHIFTER': 1, 'GEAR': gear_data})
        else:
            encoded_gear_data = self.gear_message.encode({'GEAR_SHIFTER': 0, 'GEAR': gear_data})
        message = can.Message(arbitration_id=self.gear_message.frame_id, data=encoded_gear_data)
        self.bus.send(message)

    def log_blinker_data(self, left_blinker, right_blinker):
        encoded_blinker_data = self.blinker_message.encode({'DRIVERS_DOOR_OPEN': 0, 'MAIN_ON': 0,'RIGHT_BLINKER': right_blinker,'LEFT_BLINKER': left_blinker, 'CMBS_STATES': 0})
        message = can.Message(arbitration_id=self.blinker_message.frame_id, data=encoded_blinker_data)
        self.bus.send(message)

    def log_headlight_data(self, low_beam, high_beam):
        headlight_data = 1 if (low_beam | high_beam == 1) else 0
        encoded_headlight_data = self.headlight_message.encode({'AUTO_HEADLIGHTS': 0, 'HIGH_BEAM_HOLD': 0, 'HIGH_BEAM_FLASH': 0, 'HEADLIGHTS_ON': headlight_data, 'WIPER_SWITCH': 0})
        message = can.Message(arbitration_id=self.headlight_message.frame_id, data=encoded_headlight_data)
        self.bus.send(message)

    def log_beam_data(self, low_beam, high_beam, park_lights):
        encoded_beam_data = self.beam_message.encode({'WIPERS': 0, 'LOW_BEAMS': low_beam, 'HIGH_BEAMS': high_beam, 'PARK_LIGHTS': park_lights})
        message = can.Message(arbitration_id=self.beam_message.frame_id, data=encoded_beam_data)
        self.bus.send(message)

    def log_handbrake_data(self, handbrake):
        handbrake_data = 1 if (handbrake == True) else 0
        encoded_handbrake_data = self.handbrake_message.encode({'ESP_DISABLED': 0, 'USER_BRAKE': handbrake_data, 'BRAKE_HOLD_ACTIVE': 0, 'BRAKE_HOLD_ENABLED': 0})
        message = can.Message(arbitration_id=self.handbrake_message.frame_id, data=encoded_handbrake_data)
        self.bus.send(message)

    def log_speed_data(self, speed):
        encoded_speed_data = self.speed_message.encode({'CAR_SPEED': speed})
        message = can.Message(arbitration_id=self.speed_message.frame_id, data=encoded_speed_data)
        self.bus.send(message)

    def log_offline_data(self, steer_data, throttle_data, brake_data, gear_data, manual_gear_shift, left_blinker, right_blinker, low_beam, high_beam, park_lights, handbrake,speed_data):
        self.log_steer_data(steer_data)
        self.log_throttle_brake_data(throttle_data, brake_data)
        self.log_gear_data(gear_data, manual_gear_shift)
        self.log_blinker_data(left_blinker, right_blinker)
        self.log_headlight_data(low_beam, high_beam)
        self.log_beam_data(low_beam, high_beam, park_lights)
        self.log_handbrake_data(handbrake)
        self.log_speed_data(speed_data)

    def __del__(self):
        self.bus.shutdown()
        pass

# Method to add a delay in the simulation loop
def add_delay(delay):
    current_time = timeit.default_timer()
    delay_time = current_time + delay
    while current_time < delay_time:
        current_time = timeit.default_timer()

# Method to generate a jitter array with random values between min_jitter and max_jitter
def generate_jitter_array(min_jitter, max_jitter, size):
    return [random.uniform(min_jitter, max_jitter) for _ in range(size)]

def modify_vehicle_physics(vehicle):
    physics_control = vehicle.get_physics_control()
    physics_control.use_sweeping_control = True
    vehicle.apply_physics_control(physics_control)

# Method to get the current status of the vehicle's lights and speed
def get_status(vehicle):
    # Current light state is obtained from the vehicle
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
    source = spawn_points[11] # Choose a spawn point for source vehicle
    source.location.y -= 25
    vehicle = world.try_spawn_actor(blueprint, source) # Spawn the vehicle at the source location
    world.wait_for_tick() # Wait for the world to tick before proceeding

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.006  # Lower the fixed delta seconds for more stable simulation  
    world.apply_settings(settings)

    opt_dict = {'follow_speed_limits': False, 'ignore_traffic_lights': True, 'ignore_stop_signs': True, 'ignore_vehicles': True}

    # Create an agent for the vehicle
    agent = BasicAgent(vehicle, target_speed=100, opt_dict=opt_dict)
    # agent = BehaviorAgent(vehicle, behavior='normal', opt_dict=opt_dict)
    
    # Set the destination for the agent
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
    can_handler = CAN_Data_Logger_Realtime()

    # Time Difference between ticks
    time_diff_tick = 0.004

    # Simulation time in seconds
    sim_seconds = 20

    # Calculate total iterations based on simulation time and time difference
    total_iterations = int(sim_seconds / time_diff_tick)

    # Generate jitter array for CAN data logging
    jitter_array = generate_jitter_array(0,0.0001,total_iterations*14)

    # Start simulation
    start_time = timeit.default_timer()
    try:
        for i in range(total_iterations):  
            # Get current time
            current_time = timeit.default_timer() - start_time

            timestamps.append(current_time)

            world.tick()

            # Get current control state
            control = agent.run_step()

            # Get current location of the vehicle
            location = vehicle.get_location()

            # Apply the control to the vehicle
            vehicle.apply_control(control)

            # Get the current light status and speed of the vehicle
            light_status, speed = get_status(vehicle)

            # Log the data to CAN messages
            can_handler.log_realtime_data(
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
                    jitter_array
                )

            # Log Vehicle Control Data
            vehicle_control_obj.append(control)

            # Logging vehicle location data
            vehicle_location.append(location)
   
    finally:
        print("Terminating simulation...")
        print()

        # Write data to log files
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
        
        # Destroy vehicle after simulation, else it will remain in the world
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

    # for i, transform in enumerate(spawn_points):
    #     location = transform.location + carla.Location(z=1.5)  # lift text slightly above ground
    #     world.debug.draw_string(
    #         location,
    #         str(i),
    #         draw_shadow=False,
    #         color=carla.Color(255, 0, 0),
    #         life_time=60.0,  # Show for 60 seconds
    #         persistent_lines=True
    #     )

    # Spawn vehicle
    # car_model_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
    # car_model = car_model_list[2]
    blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    blueprint.set_attribute('color', '0,0,255')
    source = spawn_points[11]
    vehicle_2 = world.try_spawn_actor(blueprint, source)
    world.wait_for_tick()
    source.location.y -= 25
    blueprint.set_attribute('color', '255,0,0')
    vehicle_1 = world.try_spawn_actor(blueprint, source)
    world.wait_for_tick()

    # Configure synchronous simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    # Lower the fixed delta seconds for more stable simulation
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
    vehicle_location_writer_1 = open('./Logs/gen_coord_1.log', 'w')
    vehicle_control_writer_1 = open('./Logs/gen_control_obj_1.log', 'w')

    vehicle_location_writer_2 = open('./Logs/gen_coord_2.log', 'w')
    vehicle_control_writer_2 = open('./Logs/gen_control_obj_2.log', 'w')

    timestamp_writer = open('./Logs/gen_timestamps.log', 'w')
    vc_timestamp_writer = open('./Logs/gen_vehicle_control_time.log', 'w')

    # Initialize lists to store data
    vehicle_control_obj_1 = []
    vehicle_control_obj_2 = []
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []
    vehicle_light_state = []
    vehicle_velocity = []
    vehicle_control_time = []

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger_Realtime()
    # can_handler_offline = CAN_Data_Logger_Offline(world)

    # Time Difference between ticks
    time_diff_tick = 0.006

    # Simulation time in seconds
    sim_time_sec = 120

    # Calculate total iterations based on simulation time and time difference
    total_iterations = int(sim_time_sec / time_diff_tick)

    # Generate jitter array for CAN data logging
    jitter_array = generate_jitter_array(0,0.0001,total_iterations*14)

    # Start simulation
    start_time = timeit.default_timer()
    can_handler.start_frame()  # Start frame for CAN data logging

    # current_time = timeit.default_timer() - start_time
    try:
        for i in range(total_iterations):  
            # Get current time
            current_time = timeit.default_timer() - start_time
            # print(f"Current Time: {current_time} seconds")

            # print(str(current_time))
            timestamps.append(current_time)

            # Tick the world to update the simulation
            world.tick()

            if agent_1.done():
                break
                new_index = random.randint(0, len(spawn_points) - 1)
                destination = spawn_points[new_index].location
                agent_1.set_destination(destination)
                agent_2.set_destination(destination)

            # Get current control state
            control_1 = agent_1.run_step()
            control_2 = agent_2.run_step()

            # Get current location of the vehicles
            current_location_1 = vehicle_1.get_location()
            current_location_2 = vehicle_2.get_location()

            #Application of benign control
            vehicle_1.apply_control(control_1)
            vehicle_2.apply_control(control_2)

            # Get the current light status and speed of the vehicles
            light_status, speed = get_status(vehicle_1)
            # vehicle_light_state.append(vehicle_1.get_light_state())
            # vehicle_velocity.append(vehicle_1.get_velocity())

            vehicle_control_apply_time = timeit.default_timer() - start_time
            vehicle_control_time.append(vehicle_control_apply_time)
            # Log the data to CAN messages for both vehicles
            can_handler.log_realtime_data(
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
                    jitter_array
                )

            #Log Vehicle Control Data  
            vehicle_control_obj_1.append(control_1)
            vehicle_control_obj_2.append(control_2)

            # Logging vehicle location data
            vehicle_location_1.append(current_location_1)
            vehicle_location_2.append(current_location_2)

            # Wait for the next tick
            delay_time = current_time + time_diff_tick
            current_time = timeit.default_timer() - start_time
            while current_time < delay_time:
                current_time = timeit.default_timer() - start_time

    finally:
        print("Terminating simulation...")
        print()

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

        #     can_handler_offline.log_offline_data(
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

        # Write data to log files
        for control in vehicle_control_obj_1:
            vehicle_control_writer_1.write(str(control) + "\n")
        vehicle_control_writer_1.close()

        for control in vehicle_control_obj_2:
            vehicle_control_writer_2.write(str(control) + "\n")
        vehicle_control_writer_2.close()
        
        # for ele in timestamps:
        #     if (ele < 0.000001):
        #         timestamp_writer.write("00.000000" + '\n')
        #     elif (ele < 10):
        #         timestamp_writer.write("0" + str(ele)[:8] + '\n')
        #     else:
        #         timestamp_writer.write(str(ele)[:9] + '\n')
        # timestamp_writer.close()

        for ele in timestamps:
            if ele < 0.000001:
                timestamp_writer.write("00.000000\n")
            elif ele < 10:
                timestamp_writer.write(f"0{ele:.6f}\n")
            else:
                timestamp_writer.write(f"{ele:.6f}\n")
        timestamp_writer.close()

        for ele in vehicle_control_time:
            if (ele < 0.000001):
                vc_timestamp_writer.write("00.000000" + '\n')
            elif (ele < 10):
                vc_timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                vc_timestamp_writer.write(str(ele)[:9] + '\n')
        vc_timestamp_writer.close()

        for location in vehicle_location_1:
            vehicle_location_writer_1.write(str(location) + "\n")
        vehicle_location_writer_1.close()

        for location in vehicle_location_2:
            vehicle_location_writer_2.write(str(location) + "\n")
        vehicle_location_writer_2.close()

        # Plotting the data
        plot_vc_time(vehicle_control_obj_1, timestamps, "./Graphs/plot_throttle_time_1.png", "./Graphs/plot_steer_time_1.png", "./Graphs/plot_brake_time_1.png")
        plot_diff(timestamps, "./Graphs/plot_timestamp_diff_gen.png")
        plot_euclid_diff(vehicle_location_1, vehicle_location_2, timestamps, "./Graphs/plot_euclid_diff_benign.png")
        plot_gen_path_only_from_list(vehicle_location_1, './Graphs/plot_gen_path_only.png', 0, total_iterations)
        plot_diff(vehicle_control_time, "./Graphs/plot_vehicle_control_time_diff.png")

        # plot_euclid_diff_single_function(vehicle_location_1, vehicle_location_2, timestamps,'./Logs_Car','./Graphs/plot_euclid_diff_gen_time.png')
        # plot_euclid_diff_by_index_only(vehicle_location_1, vehicle_location_2,'./Logs_Car', './Graphs/plot_euclid_diff_gen_index.png', 0, total_iterations)
        # plot_gen_vs_rep_paths_from_files("./Logs_Car/gen_coord_1.log", "./Logs/gen_coord_1.log", "./Graphs/benign_vs_attack_path.png", 0, total_iterations)

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        # Destroy vehicles after simulation, else they will remain in the world
        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")


if __name__ == '__main__':
    print("Starting generation...")
    generate_data_two_car()
