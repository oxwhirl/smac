import numpy as np
import re
import subprocess
import platform
from absl import logging
import math
import time
import collections
import os
import pygame
import queue

from pysc2.lib import colors
from pysc2.lib import point
from pysc2.lib.renderer_human import _Surface
from pysc2.lib import transform
from pysc2.lib import features


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def _get_desktop_size():
    """Get the desktop size."""
    if platform.system() == "Linux":
        try:
            xrandr_query = subprocess.check_output(["xrandr", "--query"])
            sizes = re.findall(
                r"\bconnected primary (\d+)x(\d+)", str(xrandr_query)
            )
            if sizes[0]:
                return point.Point(int(sizes[0][0]), int(sizes[0][1]))
        except ValueError:
            logging.error("Failed to get the resolution from xrandr.")

    # Most general, but doesn't understand multiple monitors.
    display_info = pygame.display.Info()
    return point.Point(display_info.current_w, display_info.current_h)


class StarCraft2Renderer:
    def __init__(self, env, mode):
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

        self.env = env
        self.mode = mode
        self.obs = None
        self._window_scale = 0.75
        self.game_info = game_info = self.env._controller.game_info()
        self.static_data = self.env._controller.data()

        self._obs_queue = queue.Queue()
        self._game_times = collections.deque(
            maxlen=100
        )  # Avg FPS over 100 frames.  # pytype: disable=wrong-keyword-args
        self._render_times = collections.deque(
            maxlen=100
        )  # pytype: disable=wrong-keyword-args
        self._last_time = time.time()
        self._last_game_loop = 0
        self._name_lengths = {}

        self._map_size = point.Point.build(game_info.start_raw.map_size)
        self._playable = point.Rect(
            point.Point.build(game_info.start_raw.playable_area.p0),
            point.Point.build(game_info.start_raw.playable_area.p1),
        )

        window_size_px = point.Point(
            self.env.window_size[0], self.env.window_size[1]
        )
        window_size_px = self._map_size.scale_max_size(
            window_size_px * self._window_scale
        ).ceil()
        self._scale = window_size_px.y // 32

        self.display = pygame.Surface(window_size_px)

        if mode == "human":
            self.display = pygame.display.set_mode(window_size_px, 0, 32)
            pygame.display.init()

            pygame.display.set_caption("Starcraft Viewer")
        pygame.font.init()
        self._world_to_world_tl = transform.Linear(
            point.Point(1, -1), point.Point(0, self._map_size.y)
        )
        self._world_tl_to_screen = transform.Linear(scale=window_size_px / 32)
        self.screen_transform = transform.Chain(
            self._world_to_world_tl, self._world_tl_to_screen
        )

        surf_loc = point.Rect(point.origin, window_size_px)
        sub_surf = self.display.subsurface(
            pygame.Rect(surf_loc.tl, surf_loc.size)
        )
        self._surf = _Surface(
            sub_surf,
            None,
            surf_loc,
            self.screen_transform,
            None,
            self.draw_screen,
        )

        self._font_small = pygame.font.Font(None, int(self._scale * 0.5))
        self._font_large = pygame.font.Font(None, self._scale)

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def _get_units(self):
        for u in sorted(
            self.obs.observation.raw_data.units,
            key=lambda u: (u.pos.z, u.owner != 16, -u.radius, u.tag),
        ):
            yield u, point.Point.build(u.pos)

    def get_unit_name(self, surf, name, radius):
        """Get a length limited unit name for drawing units."""
        key = (name, radius)
        if key not in self._name_lengths:
            max_len = surf.world_to_surf.fwd_dist(radius * 1.6)
            for i in range(len(name)):
                if self._font_small.size(name[: i + 1])[0] > max_len:
                    self._name_lengths[key] = name[:i]
                    break
            else:
                self._name_lengths[key] = name
        return self._name_lengths[key]

    def render(self, mode):
        self.obs = self.env._obs
        self.score = self.env.reward
        self.step = self.env._episode_steps

        now = time.time()
        self._game_times.append(
            (
                now - self._last_time,
                max(
                    1,
                    self.obs.observation.game_loop
                    - self.obs.observation.game_loop,
                ),
            )
        )

        if mode == "human":
            pygame.event.pump()

        self._surf.draw(self._surf)

        observation = np.array(pygame.surfarray.pixels3d(self.display))

        if mode == "human":
            pygame.display.flip()

        self._last_time = now
        self._last_game_loop = self.obs.observation.game_loop
        # self._obs_queue.put(self.obs)
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if mode == "rgb_array"
            else None
        )

    def draw_base_map(self, surf):
        """Draw the base map."""
        hmap_feature = features.SCREEN_FEATURES.height_map
        hmap = self.env.terrain_height * 255
        hmap = hmap.astype(np.uint8)
        if (
            self.env.map_name == "corridor"
            or self.env.map_name == "so_many_baneling"
            or self.env.map_name == "2s_vs_1sc"
        ):
            hmap = np.flip(hmap)
        else:
            hmap = np.rot90(hmap, axes=(1, 0))
        if not hmap.any():
            hmap = hmap + 100  # pylint: disable=g-no-augmented-assignment
        hmap_color = hmap_feature.color(hmap)
        out = hmap_color * 0.6

        surf.blit_np_array(out)

    def draw_units(self, surf):
        """Draw the units."""
        unit_dict = None  # Cache the units {tag: unit_proto} for orders.
        tau = 2 * math.pi
        for u, p in self._get_units():
            fraction_damage = clamp(
                (u.health_max - u.health) / (u.health_max or 1), 0, 1
            )
            surf.draw_circle(
                colors.PLAYER_ABSOLUTE_PALETTE[u.owner], p, u.radius
            )

            if fraction_damage > 0:
                surf.draw_circle(
                    colors.PLAYER_ABSOLUTE_PALETTE[u.owner] // 2,
                    p,
                    u.radius * fraction_damage,
                )
            surf.draw_circle(colors.black, p, u.radius, thickness=1)

            if self.static_data.unit_stats[u.unit_type].movement_speed > 0:
                surf.draw_arc(
                    colors.white,
                    p,
                    u.radius,
                    u.facing - 0.1,
                    u.facing + 0.1,
                    thickness=1,
                )

            def draw_arc_ratio(
                color, world_loc, radius, start, end, thickness=1
            ):
                surf.draw_arc(
                    color, world_loc, radius, start * tau, end * tau, thickness
                )

            if u.shield and u.shield_max:
                draw_arc_ratio(
                    colors.blue, p, u.radius - 0.05, 0, u.shield / u.shield_max
                )

            if u.energy and u.energy_max:
                draw_arc_ratio(
                    colors.purple * 0.9,
                    p,
                    u.radius - 0.1,
                    0,
                    u.energy / u.energy_max,
                )
            elif u.orders and 0 < u.orders[0].progress < 1:
                draw_arc_ratio(
                    colors.cyan, p, u.radius - 0.15, 0, u.orders[0].progress
                )
            if u.buff_duration_remain and u.buff_duration_max:
                draw_arc_ratio(
                    colors.white,
                    p,
                    u.radius - 0.2,
                    0,
                    u.buff_duration_remain / u.buff_duration_max,
                )
            if u.attack_upgrade_level:
                draw_arc_ratio(
                    self.upgrade_colors[u.attack_upgrade_level],
                    p,
                    u.radius - 0.25,
                    0.18,
                    0.22,
                    thickness=3,
                )
            if u.armor_upgrade_level:
                draw_arc_ratio(
                    self.upgrade_colors[u.armor_upgrade_level],
                    p,
                    u.radius - 0.25,
                    0.23,
                    0.27,
                    thickness=3,
                )
            if u.shield_upgrade_level:
                draw_arc_ratio(
                    self.upgrade_colors[u.shield_upgrade_level],
                    p,
                    u.radius - 0.25,
                    0.28,
                    0.32,
                    thickness=3,
                )

            def write_small(loc, s):
                surf.write_world(self._font_small, colors.white, loc, str(s))

            name = self.get_unit_name(
                surf,
                self.static_data.units.get(u.unit_type, "<none>"),
                u.radius,
            )

            if name:
                write_small(p, name)

            start_point = p
            for o in u.orders:
                target_point = None
                if o.HasField("target_unit_tag"):
                    if unit_dict is None:
                        unit_dict = {
                            t.tag: t
                            for t in self.obs.observation.raw_data.units
                        }
                    target_unit = unit_dict.get(o.target_unit_tag)
                    if target_unit:
                        target_point = point.Point.build(target_unit.pos)
                if target_point:
                    surf.draw_line(colors.cyan, start_point, target_point)
                    start_point = target_point
                else:
                    break

    def draw_overlay(self, surf):
        """Draw the overlay describing resources."""
        obs = self.obs.observation
        times, steps = zip(*self._game_times)
        sec = obs.game_loop // 22.4
        surf.write_screen(
            self._font_large,
            colors.green,
            (-0.2, 0.2),
            "Score: %s, Step: %s, %.1f/s, Time: %d:%02d"
            % (
                self.score,
                self.step,
                sum(steps) / (sum(times) or 1),
                sec // 60,
                sec % 60,
            ),
            align="right",
        )
        surf.write_screen(
            self._font_large,
            colors.green * 0.8,
            (-0.2, 1.2),
            "APM: %d, EPM: %d, FPS: O:%.1f, R:%.1f"
            % (
                obs.score.score_details.current_apm,
                obs.score.score_details.current_effective_apm,
                len(times) / (sum(times) or 1),
                len(self._render_times) / (sum(self._render_times) or 1),
            ),
            align="right",
        )

    def draw_screen(self, surf):
        """Draw the screen area."""
        self.draw_base_map(surf)
        self.draw_units(surf)
        self.draw_overlay(surf)
