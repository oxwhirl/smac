import numpy as np
import re
import subprocess
import platform
import os
from absl import logging
import math

from pysc2.lib import colors
from pysc2.lib import point
from pysc2.lib.renderer_human import _Surface
from pysc2.lib import transform
from pysc2.lib import features

from s2clientprotocol import raw_pb2 as sc_raw

background_rgb = (0, 255, 0)

def clamp(n, smallest, largest):
  return max(smallest, min(n, largest))

def _get_desktop_size():
  """Get the desktop size."""
  if platform.system() == "Linux":
    try:
      xrandr_query = subprocess.check_output(["xrandr", "--query"])
      sizes = re.findall(r"\bconnected primary (\d+)x(\d+)", str(xrandr_query))
      if sizes[0]:
        return point.Point(int(sizes[0][0]), int(sizes[0][1]))
    except:  # pylint: disable=bare-except
      logging.error("Failed to get the resolution from xrandr.")

  # Most general, but doesn't understand multiple monitors.
  display_info = pygame.display.Info()
  return point.Point(display_info.current_w, display_info.current_h)

class Renderer:
    def __init__(self, env, map_size, mode):
        import pygame  #initialize pygame
        self.env = env
        self.mode = mode
        self.obs = None
        self._window_scale = 0.75
        self.game_info = game_info = self.env._controller.game_info()
        self.static_data = static_data = self.env._controller.data()

        self._map_size = point.Point.build(game_info.start_raw.map_size)
        self._playable = point.Rect(
        point.Point.build(game_info.start_raw.playable_area.p0),
        point.Point.build(game_info.start_raw.playable_area.p1))

        self._name_lengths = {}

        window_size_px = self._map_size.scale_max_size(_get_desktop_size() * self._window_scale).ceil()

        self._scale = window_size_px.y // 32

        # The sub-surfaces that the various draw functions will draw to.
        self._surfaces = []
        def add_surface(surf_type, surf_loc, world_to_surf, world_to_obs, draw_fn):
            """Add a surface. Drawn in order and intersect in reverse order."""
            sub_surf = self.display.subsurface(pygame.Rect(surf_loc.tl, surf_loc.size))
            self._surfaces.append(_Surface(sub_surf, surf_type, surf_loc, world_to_surf, world_to_obs, draw_fn))

        def check_eq(a, b):
            """Used to run unit tests on the transforms."""
            assert (a - b).len() < 0.0001, "%s != %s" % (a, b)
        
        # World has origin at bl, world_tl has origin at tl.
        self._world_to_world_tl = transform.Linear(point.Point(1, -1), point.Point(0, self._map_size.y))
        
        check_eq(self._world_to_world_tl.fwd_pt(point.Point(0, 0)), point.Point(0, self._map_size.y))
        check_eq(self._world_to_world_tl.fwd_pt(point.Point(5, 10)), point.Point(5, self._map_size.y - 10))
        
        # Move the point to be relative to the camera. This gets updated per frame.
        self._world_tl_to_world_camera_rel = transform.Linear(
        offset=-self._map_size / 4)
        
        check_eq(self._world_tl_to_world_camera_rel.fwd_pt(self._map_size / 4), point.Point(0, 0))
        
        check_eq(self._world_tl_to_world_camera_rel.fwd_pt((self._map_size / 4) + point.Point(5, 10)), point.Point(5, 10))

        if mode == "human":
            pygame.init()
            pygame.display.init()

            self._font_small = pygame.font.Font(None, int(self._scale * 0.5))
            self._font_large = pygame.font.Font(None, self._scale)

            infoObject = pygame.display.Info()
            screen_size = (infoObject.current_w - 50, infoObject.current_h - 50)
            self.resolution = resolution = np.min([screen_size, map_size], axis=0)
            self.display = pygame.display.set_mode(window_size_px, 0, 32)
            canvas_resolution = (resolution[0], resolution[1])
            self.canvas = pygame.Surface(canvas_resolution)
            pygame.display.set_caption("Starcraft Viewer")

            self._feature_screen_px = point.Point(84, 84)
            main_screen_px = self._feature_screen_px
            window_size_ratio = main_screen_px
            self._feature_camera_width_world_units = 24
            
            num_feature_layers = 0
            if self.game_info.options.raw:
                num_feature_layers += 5
            if self._feature_screen_px:
                num_feature_layers += 27
                num_feature_layers += 11
            if num_feature_layers > 0:
                feature_cols = math.ceil(math.sqrt(num_feature_layers))
                feature_rows = math.ceil(num_feature_layers / feature_cols)
                features_layout = point.Point(feature_cols, feature_rows * 1.05)  # Make room for titles.
                
                # Scale features_layout to main_screen_px height so we know its width.
                features_aspect_ratio = (features_layout * main_screen_px.y / features_layout.y)
                window_size_ratio += point.Point(features_aspect_ratio.x, 0)
            
            screen_size_px = main_screen_px.scale_max_size(window_size_px)
            
            # Feature layer locations in continuous space.
            feature_world_per_pixel = (self._feature_screen_px /self._feature_camera_width_world_units)
            world_camera_rel_to_feature_screen = transform.Linear(feature_world_per_pixel, self._feature_screen_px / 2)
            
            check_eq(world_camera_rel_to_feature_screen.fwd_pt(point.Point(0, 0)),self._feature_screen_px / 2)
            check_eq(world_camera_rel_to_feature_screen.fwd_pt(point.Point(-0.5, -0.5) * self._feature_camera_width_world_units),point.Point(0, 0))
            
            self._world_to_feature_screen = transform.Chain(self._world_to_world_tl,self._world_tl_to_world_camera_rel,world_camera_rel_to_feature_screen)
            self._world_to_feature_screen_px = transform.Chain(self._world_to_feature_screen,transform.PixelToCoord())

            print("screen_size", screen_size_px)
            feature_screen_to_main_screen = transform.Linear(screen_size_px / self._feature_screen_px)
            add_surface(None, point.Rect(point.origin, screen_size_px), transform.Chain(self._world_to_feature_screen, feature_screen_to_main_screen), self._world_to_feature_screen_px, self.draw_screen)
    
    def close(self):
        import pygame
        pygame.display.quit()
        pygame.quit()
    
    def _get_units(self):
        for u in sorted(self.obs.observation.raw_data.units, key=lambda u: (u.pos.z, u.owner != 16, -u.radius, u.tag)):
            yield u, point.Point.build(u.pos)
    
    def get_unit_name(self, surf, name, radius):
        """Get a length limited unit name for drawing units."""
        key = (name, radius)
        if key not in self._name_lengths:
            max_len = surf.world_to_surf.fwd_dist(radius * 1.6)
            for i in range(len(name)):
                if self._font_small.size(name[:i + 1])[0] > max_len:
                    self._name_lengths[key] = name[:i]
                    break
            else:
                self._name_lengths[key] = name
        return self._name_lengths[key]
    
    def render(self, mode):
        import os
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
        import pygame

        pygame.event.pump()

        self.obs = self.env._obs

        print(self.obs.observation)
        
        for surf in self._surfaces:
            # Render that surface.
            surf.draw(surf)

        pygame.display.flip()
    
    def draw_base_map(self, surf):
        """Draw the base map."""
        hmap_feature = features.SCREEN_FEATURES.height_map
        hmap = hmap_feature.unpack(self.obs.observation)
        print(self.obs)
        if not hmap.any():
            hmap = hmap + 100  # pylint: disable=g-no-augmented-assignment
        hmap_color = hmap_feature.color(hmap)
        out = hmap_color * 0.6
        
        creep_feature = features.SCREEN_FEATURES.creep
        creep = creep_feature.unpack(self.obs.observation)
        creep_mask = creep > 0
        creep_color = creep_feature.color(creep)
        out[creep_mask, :] = (0.4 * out[creep_mask, :] + 0.6 * creep_color[creep_mask, :])
        
        power_feature = features.SCREEN_FEATURES.power
        power = power_feature.unpack(self.obs.observation)
        power_mask = power > 0
        power_color = power_feature.color(power)
        out[power_mask, :] = (0.7 * out[power_mask, :] + 0.3 * power_color[power_mask, :])
        
        if self._render_player_relative:
            player_rel_feature = features.SCREEN_FEATURES.player_relative
            player_rel = player_rel_feature.unpack(self.obs.observation)
            player_rel_mask = player_rel > 0
            player_rel_color = player_rel_feature.color(player_rel)
            out[player_rel_mask, :] = player_rel_color[player_rel_mask, :]
            
        visibility = features.SCREEN_FEATURES.visibility_map.unpack(self.obs.observation)
        visibility_fade = np.array([[0.5] * 3, [0.75]*3, [1]*3])
        out *= visibility_fade[visibility]
        
        surf.blit_np_array(out)

    def draw_units(self, surf):
        """Draw the units and buildings."""
        unit_dict = None  # Cache the units {tag: unit_proto} for orders.
        tau = 2 * math.pi

        for u, p in self._get_units():
            fraction_damage = clamp((u.health_max - u.health) / (u.health_max or 1), 0, 1)
            if u.display_type == sc_raw.Placeholder:
                surf.draw_circle(colors.PLAYER_ABSOLUTE_PALETTE[u.owner] // 3, p, u.radius)
            else:
                surf.draw_circle(colors.PLAYER_ABSOLUTE_PALETTE[u.owner], p, u.radius)

                if fraction_damage > 0:
                    surf.draw_circle(colors.PLAYER_ABSOLUTE_PALETTE[u.owner] // 2, p, u.radius * fraction_damage)
                surf.draw_circle(colors.black, p, u.radius, thickness=1)

                if self.static_data.unit_stats[u.unit_type].movement_speed > 0:
                    surf.draw_arc(colors.white, p, u.radius, u.facing - 0.1, u.facing + 0.1, thickness=1)

                def draw_arc_ratio(color, world_loc, radius, start, end, thickness=1):
                    surf.draw_arc(color, world_loc, radius, start * tau, end * tau, thickness)
                
                if u.shield and u.shield_max:
                    draw_arc_ratio(colors.blue, p, u.radius - 0.05, 0, u.shield / u.shield_max)
                
                if u.energy and u.energy_max:
                    draw_arc_ratio(colors.purple * 0.9, p, u.radius - 0.1, 0, u.energy / u.energy_max)
                if 0 < u.build_progress < 1:
                    draw_arc_ratio(colors.cyan, p, u.radius - 0.15, 0, u.build_progress)
                elif u.orders and 0 < u.orders[0].progress < 1:
                    draw_arc_ratio(colors.cyan, p, u.radius - 0.15, 0, u.orders[0].progress)
                if u.buff_duration_remain and u.buff_duration_max:
                    draw_arc_ratio(colors.white, p, u.radius - 0.2, 0, u.buff_duration_remain / u.buff_duration_max)
                if u.attack_upgrade_level:
                    draw_arc_ratio(self.upgrade_colors[u.attack_upgrade_level], p, u.radius - 0.25, 0.18, 0.22, thickness=3)
                if u.armor_upgrade_level:
                    draw_arc_ratio(self.upgrade_colors[u.armor_upgrade_level], p, u.radius - 0.25, 0.23, 0.27, thickness=3)
                if u.shield_upgrade_level:
                    draw_arc_ratio(self.upgrade_colors[u.shield_upgrade_level], p, u.radius - 0.25, 0.28, 0.32, thickness=3)
                
                def write_small(loc, s):
                    surf.write_world(self._font_small, colors.white, loc, str(s))
                
                name = self.get_unit_name(surf, self.static_data.units.get(u.unit_type, "<none>"), u.radius)
                
                if name:
                    write_small(p, name)
    def draw_effects(self, surf):
        """Draw the effects."""
        for effect in self.obs.observation.raw_data.effects:
            print(effect)
            color = [
                colors.effects[effect.effect_id],
                colors.effects[effect.effect_id],
                colors.PLAYER_ABSOLUTE_PALETTE[effect.owner],
                ]
        
            name = self.get_unit_name(surf, features.Effects(effect.effect_id).name, effect.radius)
            for pos in effect.pos:
                p = point.Point.build(pos)
                # pygame alpha transparency doesn't work, so just draw thin circles.
                for r in range(1, int(effect.radius * 3)):
                    surf.draw_circle(color[r % 3], p, r / 3, thickness=2)
                if name:
                    surf.write_world(self._font_small, colors.white, p, name)

    def draw_screen(self, surf):
        """Draw the screen area."""
        # self.draw_base_map(surf)
        # self.draw_effects(surf)
        self.draw_units(surf)



        

