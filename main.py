import sys

import matplotlib.pyplot as plt
import pandas as pd
import pygame
import numpy as np
from pandas.core.window import Window
from pygame.display import update
from pygame.draw_py import Point
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from typing import Iterable, Tuple
from collections import deque

# constants
CONST_G = 1.
DELTA_T = 0.01
FLOAT_MAX = sys.float_info.max


def euclid_dist(p1: 'PointMass', p2: 'PointMass', dim=2) -> np.float32:
    return np.linalg.norm(p1.position - p2.position, ord=dim)


class PointMass:

    class Trace:

        def __init__(self, outer: 'PointMass', color: list[int] = [255, 255, 255, 0]):
            self.list = deque(maxlen=75)

            self.outer = outer
            self.color = color

            self.window = self.outer.window

        def update(self):
            win_h, win_v = self.window.get_size()
            scale = win_h / 4

            x = self.outer.position[0] * scale + win_h / 2
            y = win_v - (self.outer.position[1] * scale + win_v / 2)

            self.list.appendleft((x, y))

        def draw(self):
            if len(self.list) > 1:
                pygame.draw.lines(self.window, self.color, False, self.list, 1)

    def __init__(self, window, mass: np.float32,
                 position: np.ndarray[np.float32] = np.array([0., 0.]),
                 velocity: np.ndarray[np.float32] = np.array([0., 0.]),
                 color: list[int] = [255, 255, 255, 255],
                 proximity_coloring=True):

        self.window = window

        self.mass = mass
        self.position = position
        self.velocity = velocity

        self.color = color
        self.trace = self.Trace(self)  # funky ahh line

        self.min_prox = FLOAT_MAX
        self.running_max = 0.

        self.prox_bool = proximity_coloring

    def compute_accelerations(self, bodies: Iterable['PointMass'], softening=0.2) -> np.ndarray[np.float32]:
        a_vec: np.ndarray[np.float32] = np.array([0., 0.])

        self.min_prox = FLOAT_MAX

        for body in bodies:
            if body is not self:
                # direction vector towards each body
                dist = euclid_dist(self, body)

                # update proximity value with minimum of distances
                self.min_prox = min(self.min_prox, dist)
                self.running_max = max(self.running_max, dist)

                # acceleration magnitude towards each body
                accel_mag = (CONST_G * body.mass * (body.position - self.position)) / np.maximum(dist ** 3,
                                                                                                 softening ** 3)

                # add to total acceleration
                a_vec += accel_mag

        return a_vec

    def update(self, bodies: Iterable['PointMass'], softening=0.2) -> np.ndarray:

        a_vec: np.ndarray[np.float32] = self.compute_accelerations(bodies, softening=softening)

        # other guy's method
        # update position
        self.position += self.velocity * DELTA_T + 0.5 * a_vec * DELTA_T ** 2
        new_a_vec: np.ndarray[np.float32] = self.compute_accelerations(bodies, softening=softening)
        # update velocity
        self.velocity += 0.5 * (a_vec + new_a_vec) * DELTA_T

        # normal
        # self.velocity += a_vec * DELTA_T
        # self.position += self.velocity * DELTA_T

        # update trace
        self.trace.update()

        # update color
        if self.prox_bool:
            self.update_color_by_prox()

    def draw(self, *args):
        win_h, win_v = self.window.get_size()
        scale = win_h / 4

        x = self.position[0] * scale + win_h / 2
        y = win_v - (self.position[1] * scale + win_v / 2)

        self.trace.draw()
        pygame.draw.circle(self.window, self.color, (x, y), 5.5)

        for bodies in args:
            for body in bodies:
                if body is not self:

                    x1 = self.position[0] * scale + win_h / 2
                    y1 = win_v - (self.position[1] * scale + win_v / 2)

                    x2 = body.position[0] * scale + win_h / 2
                    y2 = win_v - (body.position[1] * scale + win_v / 2)

                    pygame.draw.line(self.window, (255, 255, 255, 64), (x1, y1), (x2, y2), 1)

    def update_color_by_prox(self):
        # goal: update the alpha value of the body based on the current min_prox value.

        # if it's close to running max * some dampening val, then it gets darker
        norm = np.clip(self.running_max * 0.75 - self.min_prox / 0.65, 0.075, 1.)  # normalize and reverse

        self.color = (255, 255, 255, int(norm * 255))
        self.trace.color = (255, 255, 255, int(norm * 255) // 2)


def make_state(d, vx, vy, window):
    # states of each body
    middle = np.array([0, 0, vx, vy])
    left = np.array([d, 0, -middle[2] / 2, -middle[3] / 2])
    right = np.array([-d, 0, -middle[2] / 2, -middle[3] / 2])

    all_states = [middle, left, right]

    return [PointMass(window, mass=1, position=state[:2], velocity=state[2:], proximity_coloring=True) for _, state in enumerate(all_states)]


def main_pygame(d, vx, vy):
    class Canvas:
        def __init__(self):
            self.canvasX = 0
            self.canvasY = 0

        def move(self, dx, dy):
            self.canvasX += dx
            self.canvasY += dy

    canvas = Canvas()
    WIN = pygame.display.set_mode((700, 700))
    s = pygame.Surface((700, 700), flags=pygame.SRCALPHA)

    bodies = make_state(d, vx, vy, s)

    # for i in range(steps):
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Draw background
        WIN.fill((10, 10, 10))
        s.fill((10, 10, 10))

        # Draw objects
        com = PointMass(s, 1., proximity_coloring=False)
        pos = np.array([0., 0.])

        for i, body in enumerate(bodies):
            body.update(bodies)
            # body.draw(bodies + [com])
            body.draw()

            pos += body.position

        com.position = pos / 3

        # print(com.position)

        # com.draw()

        WIN.blit(s, (0, 0))
        pygame.display.flip()
        pygame.time.delay(10)


main_pygame(1., 1., 1.)

# bodies = [PointMass(WIN, mass=1, position=np.array([0.4, 0.]), velocity=np.array([0., 0.1]), color=(255, 0, 0)),
#           PointMass(WIN, mass=1, position=np.array([-1., 0.]), velocity=np.array([0., -0.1]), color=(0, 255, 0)),
#           PointMass(WIN, mass=1, position=np.array([0., 1.]), velocity=np.array([-0.1, 0.]), color=(0, 0, 255))]
