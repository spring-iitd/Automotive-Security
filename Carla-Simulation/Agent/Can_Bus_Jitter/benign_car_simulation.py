#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Starting Simulation...
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import re
import sys
import weakref
import timeit
import random
import can
import cantools
import subprocess
import time
import timeit
from plot_graphs import plot_diff, plot_graph, plot_x_time, plot_y_time, plot_vc_time, plot_path_diff, plot_euclid_diff, plot_trajectories_with_customization, plot_can_log_diff, plot_euclid_diff_single_function
import pdb
from datetime import datetime

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

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

import carla
from carla import ColorConverter as cc
from agents.navigation.custom_agent import CustomAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args, start_index, x_adj, y_adj, z_adj):
        """Constructor method"""
        self._args = args
        self.world = carla_world                # non-pickalable
        try:
            self.map = self.world.get_map()     # non-pickalable 
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.start_index = start_index
        self.x_adj = x_adj
        self.y_adj = y_adj
        self.z_adj = z_adj
        self.restart(args)
        self.previous_weather = carla_world.get_weather()

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random car_model.
        car_model_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        car_model = car_model_list[2]
        car_model.set_attribute('role_name', 'hero')
        if car_model.has_attribute('color'):
            color = '0,0,0'
            car_model.set_attribute('color', color)

        # # Get the blueprint library
        # blueprint_library = self.world.get_blueprint_library()

        # # Find the Cybertruck blueprint
        # car_model = blueprint_library.find('vehicle.bh.crossbike')
        # car_model = blueprint_library.find('vehicle.carlamotors.european_hgv')
        # car_model = blueprint_library.find('vehicle.tesla.model3')

        car_model.set_attribute('role_name', 'hero')
        # if car_model.has_attribute('color'):
        #     color = '255, 0, 0'
        #     car_model.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(car_model, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            car_start_pos = spawn_points[self.start_index]
            car_start_pos.location.x += self.x_adj
            car_start_pos.location.y += self.y_adj
            car_start_pos.location.z += self.z_adj
            self.player = self.world.try_spawn_actor(car_model, car_start_pos)
            self.modify_vehicle_physics(self.player)

        self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def set_night_weather(self):
        """Set the weather to night"""
        weather = carla.WeatherParameters(
            sun_altitude_angle=-90.0,  # This sets the sun below the horizon, simulating night
            cloudiness=0.0,  # Optional: Adjust cloudiness as needed
            precipitation=0.0,  # Optional: Adjust precipitation as needed
            fog_density=0.0,  # Optional: Adjust fog as needed
            wind_intensity=0.0,  # Optional: Adjust wind as needed
            fog_falloff=0.0,  # Optional: Adjust fog falloff as needed
            wetness=0.0  # Optional: Adjust wetness as needed
        )
        self.world.set_weather(weather)
        self.hud.notification('Weather: Night')

    def get_light_status(self):
        """Get the light status"""
        current_lights = self.player.get_light_state()

        # Mapping light states to their respective variables
        light_states = {
            "left_blinker_set": carla.VehicleLightState.LeftBlinker,
            "right_blinker_set": carla.VehicleLightState.RightBlinker,
            "low_beam_set": carla.VehicleLightState.LowBeam,
            "high_beam_set": carla.VehicleLightState.LowBeam,
            "park_lights_set": carla.VehicleLightState.Position,
        }

        # Initialize variables dynamically
        light_status = {key: 1 if current_lights & value else 0 for key, value in light_states.items()}

        # Access the values
        left_blinker_set = light_status["left_blinker_set"]
        right_blinker_set = light_status["right_blinker_set"]
        low_beam_set = light_status["low_beam_set"]
        high_beam_set = light_status["high_beam_set"]
        park_lights_set = light_status["park_lights_set"]

        return left_blinker_set, right_blinker_set, low_beam_set, high_beam_set, park_lights_set

    def reset_weather(self):
        """Reset the weather to its previous state"""
        if hasattr(self, 'previous_weather'):
            self.world.set_weather(self.previous_weather)
            self.hud.notification('Weather: Reset')
        else:
            self.hud.notification('No previous weather stored')

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']

        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]

        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================

class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        car_model = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(car_model, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        car_model = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(car_model, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- CAN -----------------------------------------------------------------------
# ==============================================================================

class CAN_Data_Logger(object):
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

    def log_data(self, steer_data, throttle_data, brake_data, gear_data, manual_gear_shift, left_blinker, right_blinker, low_beam, high_beam, park_lights, handbrake,speed_data):
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

# ==============================================================================
# -- Game Loop -----------------------------------------------------------------
# ==============================================================================
def game_loop(args):

    """
    Main loop of the simulation. It handles updating all the HUD information,

    ticking the agent and, if needed, the world.
    """

    coord_writer_1 = open('./Logs/gen_coord_1.log', 'w')
    vc_writer_1 = open("./Logs/gen_control_obj_1.log", "w")

    coord_writer_2 = open('./Logs/gen_coord_2.log', 'w')
    vc_writer_2 = open("./Logs/gen_control_obj_2.log", "w")

    # timediff_writer = open('./Logs/time_diff.log', 'w')
    timestamp_writer = open('./Logs/gen_timestamps.log', 'w')
    destination_path_writer = open('./Logs/destination_path.log', 'w')

    gen_control_obj_1, gen_coord_1 = [], []
    gen_control_obj_2, gen_coord_2 = [], []
    gen_time_diff = []
    gen_timestamps = []
    destination_path = []
    vehicle_light_state = []
    vehicle_velocity = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        pygame.init()

        # get_world() fetches the current map from the server
        # sim_world = client.get_world()
        sim_world = client.load_world('Town05_Opt', reset_settings=True, map_layers= carla.MapLayer.NONE)
        sim_world.load_map_layer(carla.MapLayer.Buildings)
        sim_world.load_map_layer(carla.MapLayer.Ground)

        # load_world() can be used to load map according to user requirement
        # sim_world_1 = client_1.load_world('Town10HD_Opt')
        # sim_world = client_1.load_world('Town05_Opt')
        # sim_world.unload_map_layer(carla.MapLayer.All)  # Unload buildings layer
        # sim_world.load_map_layer(carla.MapLayer.All)  # Load all map layers
        # sim_world.unload_map_layer(carla.MapLayer.All)  # Unload all map layers

        sim_time_sec = 22.5 # Duration of simulation in seconds
        source_location = 11
        hud = HUD(args.width, args.height)
        world_1 = World(sim_world, hud, args, source_location, 0, -25, 0)
        world_2 = World(sim_world, hud, args, source_location, 0, 0, 0)
        destination_path.append(source_location)
      
        clock = pygame.time.Clock()
        
        opt_dict = {'follow_speed_limits': False, 'ignore_traffic_lights': True, 'ignore_stop_signs': True, 'ignore_vehicles': True}

        # agent_1 = BehaviorAgent(world_1.player, behavior='normal', opt_dict=opt_dict)
        # agent_2 = BehaviorAgent(world_2.player, behavior='normal', opt_dict=opt_dict)
        agent_1 = BasicAgent(world_1.player,100, opt_dict)
        agent_2 = BasicAgent(world_2.player,100, opt_dict)

        # Set the agent destination
        spawn_points = world_1.map.get_spawn_points()
        new_index = random.randint(0, len(spawn_points)-1)
        new_index = 20
        destination = spawn_points[new_index].location
        agent_1.set_destination(destination)
        agent_2.set_destination(destination)
        destination_path.append(new_index)

        settings = sim_world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.01 # Time step for synchronous mode
        sim_world.apply_settings(settings)

        can_handler = CAN_Data_Logger(world_1)
        
        index = 0
        path = []
        
        night_mode_set = False 
        startTime = timeit.default_timer() 
        while True:
            # Control number of frames per second
            clock.tick_busy_loop(80) 
          
            # Tick the server and wait for a frame
            sim_world.tick()

            # Get the controls for both agents
            control_1 = agent_1.run_step()
            control_2 = agent_2.run_step()

            # Get the current location of both vehicles    
            current_location_1 = world_1.player.get_location()
            current_location_2 = world_2.player.get_location()
            
            #Application of benign control
            world_1.player.apply_control(control_1)
            world_2.player.apply_control(control_2)

            # Get the current simulation time
            current_timestamp = timeit.default_timer()-startTime

            # Check if the simulation time has reached the specified limit
            if current_timestamp >= sim_time_sec: break

            # # Check if night mode is set and apply the light state
            if not night_mode_set and current_timestamp >= sim_time_sec / 2:
                current_lights = world_1.player.get_light_state()
                world_1.player.set_light_state(carla.VehicleLightState(current_lights | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position))
                # world_2.player.set_light_state(carla.VehicleLightState(current_lights | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position))
                night_mode_set = True  # Set the flag to True after execution
                
            # # Get the current light status
            # left_blinker_set, right_blinker_set, low_beam_set, high_beam_set, park_lights_set = world_1.get_light_status()
            # current_lights = world_1.player.get_light_state()

            # # Mapping light states to their respective variables
            # light_states = {
            #     "left_blinker_set": carla.VehicleLightState.LeftBlinker,
            #     "right_blinker_set": carla.VehicleLightState.RightBlinker,
            #     "low_beam_set": carla.VehicleLightState.LowBeam,
            #     "high_beam_set": carla.VehicleLightState.LowBeam,
            #     "park_lights_set": carla.VehicleLightState.Position,
            # }

            # # Initialize variables dynamically
            # light_status = {key: 1 if current_lights & value else 0 for key, value in light_states.items()}

            # # Access the values
            # left_blinker_set = light_status["left_blinker_set"]
            # right_blinker_set = light_status["right_blinker_set"]
            # low_beam_set = light_status["low_beam_set"]
            # high_beam_set = light_status["high_beam_set"]
            # park_lights_set = light_status["park_lights_set"]

            # Get the current light state
            light_state = world_1.player.get_light_state()
            vehicle_light_state.append(light_state)

            # # Get the current velocity of vehicle_1
            velocity_1 = world_1.player.get_velocity()
            vehicle_velocity.append(velocity_1)
            
            # Logging CAN data
            # can_handler.log_data(control_1.steer, control_1.throttle, control_1.brake, control_1.gear, control_1.manual_gear_shift, left_blinker_set, right_blinker_set,low_beam_set, high_beam_set, park_lights_set,control_1.hand_brake)

            # Logging vehicle data
            gen_timestamps.append(current_timestamp)
            gen_coord_1.append(current_location_1)
            gen_coord_2.append(current_location_2)
            gen_control_obj_1.append(control_1)
            gen_control_obj_2.append(control_2)

            # delay_time = current_timestamp + 0.010
            # current_timestamp = timeit.default_timer() - startTime
            # while current_timestamp < delay_time:
            #     current_timestamp = timeit.default_timer() - startTime
            
    finally:
        print("Terminating simulation...")
        print()

        # Candump Control Object to CAN
        for control, light_state, velocity in zip(gen_control_obj_1, vehicle_light_state, vehicle_velocity):
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

        for ele in gen_control_obj_1:
            vc_writer_1.write(str(ele) + '\n')
        vc_writer_1.close()

        for ele in gen_control_obj_2:
            vc_writer_2.write(str(ele) + '\n')
        vc_writer_2.close()

        for ele in gen_coord_1:
            coord_writer_1.write(str(ele) + '\n')
        coord_writer_1.close()

        for ele in gen_coord_2:
            coord_writer_2.write(str(ele) + '\n')
        coord_writer_2.close()

        for ele in gen_timestamps:
            if (ele < 10):
                timestamp_writer.write("0" + str(ele)[:8] + '\n')
            else:
                timestamp_writer.write(str(ele)[:9] + '\n')

        for ele in destination_path:
            destination_path_writer.write(str(ele) + '\n')
        destination_path_writer.close()

        # for ele in gen_time_diff:
        #     if ele < 10:
        #         timediff_writer.write("0" + str(ele)[:8] + '\n')
        #     else:
        #         timediff_writer.write(str(ele)[:9] + '\n')

        # plot_x_time(gen_coord_1, gen_timestamps, "./Graphs/plot_x_coordinate_time_1.png")
        # plot_y_time(gen_coord_1, gen_timestamps, "./Graphs/plot_y_coordinate_time_1.png")
        plot_vc_time(gen_control_obj_1, gen_timestamps, "./Graphs/plot_throttle_time_1.png", "./Graphs/plot_steer_time_1.png", "./Graphs/plot_brake_time_1.png")
        
        # plot_x_time(gen_coord_2, gen_timestamps, "./Graphs/plot_x_coordinate_time_2.png")
        # plot_y_time(gen_coord_2, gen_timestamps, "./Graphs/plot_y_coordinate_time_2.png")
        # plot_vc_time(gen_control_obj_2, gen_timestamps, "./Graphs/plot_throttle_time_2.png", "./Graphs/plot_steer_time_2.png", "./Graphs/plot_brake_time_2.png")

        plot_diff(gen_timestamps, "./Graphs/plot_timestamp_diff_gen.png")
        # plot_path_diff(gen_coord_1, gen_coord_2, gen_timestamps, "./Graphs/plot_path_diff.png")
        # plot_euclid_diff(gen_coord_1, gen_coord_2, gen_timestamps, "./Graphs/plot_euclid_diff.png")
        # plot_trajectories_with_customization(gen_coord_1, gen_coord_2, gen_timestamps,"./Graphs/plot_trajectories_gen.png")
        
        # plot_graph(gen_time_diff, "./Graphs/plot_time_diff.png")
        plot_can_log_diff('./Logs/can_data_logs.log', './Graphs/plot_can_timestamp_diff_1.png')
        # plot_euclid_diff_single_function(gen_coord_1, gen_coord_2, gen_timestamps,'./Logs_Benign', './Graphs/plot_euclid_diff_gen.png')
        plot_euclid_diff(gen_coord_1, gen_coord_2, gen_timestamps, "./Graphs/plot_euclid_diff_benign.png")
        # plot_x1_x2(gen_coord_1, gen_coord_2, gen_timestamps, "./Graphs/plot_x1_x2.png")
        
        if world_1 is not None:
            world_1.destroy()

        if world_2 is not None:
            world_2.destroy()

        settings.synchronous_mode = False
        sim_world.apply_settings(settings)

        # client_1.load_world('Town05_Opt')

        print("Simulation finished...")

        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():

    """
    Main method
    """

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-d', '--debug',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()