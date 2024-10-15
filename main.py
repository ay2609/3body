import matplotlib.pyplot as plt
import pandas as pd
import pygame
import numpy as np
from pygame.draw_py import Point
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from typing import Iterable

# constants
CONST_G = 1.
DELTA_T = 0.01


def euclid_dist(p1: 'PointMass', p2: 'PointMass', dim=2):
    return np.linalg.norm(p1.position - p2.position, ord=dim)


class PointMass:
    def __init__(self, mass: np.float32,
                 position: np.ndarray[np.float32] = np.array([0., 0.]),
                 velocity: np.ndarray[np.float32] = np.array([0., 0.])):

        self.mass = mass
        self.position = position
        self.velocity = velocity

    def compute_accelerations(self, bodies: Iterable['PointMass'], softening=0.2) -> np.ndarray[np.float32]:
        a_vec: np.ndarray[np.float32] = np.array([0., 0.])

        for body in bodies:
            if body is not self:
                # direction vector towards each body
                dist = euclid_dist(self, body)

                # acceleration magnitude towards each body
                accel_mag = (CONST_G * body.mass * (body.position - self.position)) / np.maximum(dist ** 3,
                                                                                                 softening ** 3)

                # add to total acceleration
                a_vec += accel_mag

        return a_vec

    def update(self, bodies: Iterable['PointMass'], softening=0.2) -> np.ndarray:

        a_vec: np.ndarray[np.float32] = self.compute_accelerations(bodies, softening=softening)

        # update position
        self.position += self.velocity * DELTA_T + 0.5 * a_vec * DELTA_T**2

        new_a_vec: np.ndarray[np.float32] = self.compute_accelerations(bodies, softening=softening)

        # update velocity
        self.velocity += 0.5 * (a_vec + new_a_vec) * DELTA_T

    def draw(self, window):
        win_h, win_v = window.get_size()

        scale = win_h/4
        x = self.position[0] * scale + win_h/2
        y = win_v - (self.position[1] * scale + win_v/2)
        pygame.draw.circle(window, (225, 225, 225), (x, y), 4)


def main_pygame():
    class Canvas:
        def __init__(self):
            self.canvasX = 0
            self.canvasY = 0

        def move(self, dx, dy):
            self.canvasX += dx
            self.canvasY += dy

    canvas = Canvas()
    WIN = pygame.display.set_mode((700, 700))

    steps = 10000
    bodies = [PointMass(mass=1, position=np.array([0.4, 0.]), velocity=np.array([0., 0.1])),
              PointMass(mass=1, position=np.array([-1., 0.]), velocity=np.array([0., -0.1])),
              PointMass(mass=1, position=np.array([0., 1.]), velocity=np.array([-0.1, 0.]))]

    for i in range(steps):  # number of integration steps?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Draw background
        WIN.fill((10, 10, 10))

        # Draw objects
        for i, body in enumerate(bodies):
            body.update(bodies)
            body.draw(WIN)

        pygame.display.flip()
        pygame.time.delay(10)


main_pygame()
