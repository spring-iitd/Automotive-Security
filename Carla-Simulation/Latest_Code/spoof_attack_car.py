import glob
import os
import queue
import random
import re
import sys
import time
import timeit

import pygame
import carla
import can
import cantools
from plot_graphs import plot_diff, plot_euclid_diff, plot_euclid_diff_by_index_only, plot_euclid_diff_single_function, plot_gen_path_only_from_list, plot_gen_vs_rep_paths_from_files, plot_spoof_timeline, plot_vc_index, plot_vc_time

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

    def log_dummy_data_1(self):
        message = can.Message(arbitration_id=0x00000170, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)

    def log_dummy_data_2(self):
        message = can.Message(arbitration_id=0x00000202, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)
    
    def log_dummy_data_3(self):
        message = can.Message(arbitration_id=0x0000018f, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)

    def log_dummy_data_4(self):
        message = can.Message(arbitration_id=0x00000430, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)
    
    def log_dummy_data_5(self):
        message = can.Message(arbitration_id=0x000001f1, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)
    
    def log_dummy_data_6(self):
        message = can.Message(arbitration_id=0x000004b1, data=[0x00] * 8, is_extended_id=True)
        self.bus.send(message)
        
    def log_data(self, index, time_diff_tick, steer_data, throttle_data, brake_data, gear_data,
             manual_gear_shift, left_blinker, right_blinker, low_beam, high_beam,
             park_lights, handbrake, speed_data, jitter_array):

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

def generate_spoof_data_two_car():
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
    vehicle_location_writer_1 = open('./Logs_26.0_1/gen_coord_1.log', 'w')
    vehicle_control_writer_1 = open("./Logs_26.0_1/gen_control_obj_1.log", "w")

    vehicle_location_writer_2 = open('./Logs_26.0_1/gen_coord_2.log', 'w')
    vehicle_control_writer_2 = open("./Logs_26.0_1/gen_control_obj_2.log", "w")

    timestamp_writer = open('./Logs_26.0_1/gen_timestamps.log', 'w')
    spoof_timestamp_writer = open('./Logs_26.0_1/spoof_timestamp.log', 'w')
    jitter_array_writer = open('./Logs_26.0_1/jitter_array.log', 'w')
    vc_timestamp_writer = open('./Logs_26.0_1/gen_vehicle_control_time.log', 'w')

    # Initialize lists to store data
    vehicle_control_obj_1 = []
    vehicle_control_obj_2 = []
    timestamps = []
    vehicle_location_1 = []
    vehicle_location_2 = []
    spoof_timestamp = []
    vehicle_control_time = []

    # Initialize CAN data logger
    can_handler = CAN_Data_Logger()

    time_diff_tick = 0.006

    sim_time_sec = 36 # Duration of simulation in seconds
    total_iterations = int(sim_time_sec / time_diff_tick)

    jitter_array = generate_jitter_array(0,100,(total_iterations+1000)*14)

    spoof_mode = False
    count_spoof = 0
    num_spoof_msgs = random.randint(0,150)
    num_spoof_msgs = 1
    print("Number spoof: ", num_spoof_msgs)
    spoof_delay = random.uniform(0.0, 0.0050)
    spoof_delay = 0.0025
    print("Amount of Delay: ", spoof_delay) 

    # Start simulation
    start_time = timeit.default_timer()
    can_handler.start_frame()
    try:
        for i in range(total_iterations):  

            # Get current time
            current_time = timeit.default_timer() - start_time

            timestamps.append(current_time)
        
            world.tick()

            # Get current control state
            control_1 = agent_1.run_step()
            control_2 = agent_2.run_step()

            current_location_1 = vehicle_1.get_location()
            current_location_2 = vehicle_2.get_location()

            vehicle_control_apply_time = timeit.default_timer() - start_time
            vehicle_control_time.append(vehicle_control_apply_time)

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
                    jitter_array
                )

            #Log Vehicle Control Data  
            vehicle_control_obj_1.append(control_1)
            vehicle_control_obj_2.append(control_2)

            # Spoofing Attack in a specific time range
            if spoof_mode or (current_time >= 26.0 and current_time <= 26.006):
                spoof_mode = True
                count_spoof += 1
                # Check for number of benign timestamp to attack
                if count_spoof >= num_spoof_msgs:
                    spoof_mode = False

                # Add delay to the spoofed control
                delay_time = current_time + spoof_delay
                current_timestamp = timeit.default_timer()-start_time
                while current_timestamp < delay_time:
                    current_timestamp = timeit.default_timer()-start_time

                # Apply the spoofed control to the first vehicle in the current tick
                count = 0
                control = carla.VehicleControl(throttle=control_1.throttle, steer=1.0, brake=control_1.brake, gear=0)
                while count < 1:
                    current_timestamp = timeit.default_timer()-start_time
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
                            jitter_array
                        )
                   
                    vehicle_control_obj_1.append(control)

                    count += 1

            # Logging vehicle location data
            vehicle_location_1.append(current_location_1)
            vehicle_location_2.append(current_location_2)

            delay_time = current_time + time_diff_tick
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
            if (ele < 0.0001):
                timestamp_writer.write("00.000000" + '\n')
            elif (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')
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

        for ele in spoof_timestamp:
            if (ele < 10):
                spoof_timestamp_writer.write("0" + str(ele)[:11] + '\n')
            else:
                spoof_timestamp_writer.write(str(ele)[:12] + '\n')
        spoof_timestamp_writer.close()

        for ele in jitter_array:
            jitter_array_writer.write(str(ele) + '\n')
        jitter_array_writer.close()

        plot_vc_time(vehicle_control_obj_1, timestamps, "./Graphs_26.0_1/plot_throttle_time_1.png", "./Graphs_26.0_1/plot_steer_time_1.png", "./Graphs_26.0_1/plot_brake_time_1.png")
        plot_euclid_diff_by_index_only(vehicle_location_1, vehicle_location_2,'./Logs', './Graphs_26.0_1/plot_euclid_diff_gen_index.png', 0, total_iterations)
        plot_gen_vs_rep_paths_from_files("./Logs/gen_coord_1.log", "./Logs_26.0_1/gen_coord_1.log", "./Graphs_26.0_1/benign_vs_attack_path.png", 0, total_iterations)
        plot_spoof_timeline(timestamps, spoof_timestamp, './Graphs_26.0_1/plot_spoof_timeline_combined.png')

        # Rest simulation settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        vehicle_1.destroy()
        vehicle_2.destroy()

        print("Generation completed and data logged.")


if __name__ == '__main__':
    print("Starting generation...")
    generate_spoof_data_two_car()
