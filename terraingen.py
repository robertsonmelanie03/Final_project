import pygame as pg
import random
import numpy as np
import pdb
import matplotlib.pyplot as ply
from hkb_diamondsquare import DiamondSquare as DS

MAX = 256
MIN = 1

# Terrain will be split into 5 classes
#Mountain
#Hill
#Forest
#Lowland
#Water

class Block():
    def __init__(self, t, pos):
        self.pos = pos
        coin = random.random()
                # Basic color mapping

        if t >= 1: 
            self.color = (0, 0, 255)
        if t >= 1.2:
            self.color = (191, 170, 118)
        if t >= 3.5:
            self.color = (80, 200, 120)
        if t >= 4.5:
            self.color = (123, 103, 103)
        if t >= 5.35:
            self.color = (255, 255, 255)
      
       
    def draw(self, SCREEN):
        with pg.PixelArray(SCREEN) as pixel_array:
            pixel_array[self.pos[0]][self.pos[1]] = self.color





def generate(N, seed_):
    map1 = DS.diamond_square(shape=(2**N + 1, 2**N + 1), 
                         min_height=1, 
                         max_height=8,
                         roughness=0.75, random_seed=seed_)

    terrain = [[0 for _ in range(2**N + 1)] for _ in range(2**N + 1)]

    for x in range(2**N + 1):
        for y in range(2**N + 1):
            val = map1[x][y]
            pos = [x, y]
            terrain[x][y] = Block(val, pos)
    
    return terrain




class Simulation:
    def __init__(self):
        N = 9
        self.map_size = 2**N + 1
        self.SCREEN = pg.display.set_mode((2**N + 1, 2**N + 1))
        seed = 34
        #define grid of pixels
        self.grid = generate(N, seed)

        
    

    def run_sim(self):

        #framerate clock
        CLOCK = pg.time.Clock()
        RUNNING = True
        self.cur_frame = 0
        #run the program
        while RUNNING:

            #generate background
            CLOCK.tick(30)
            bg_color = (247, 241, 222)
            self.SCREEN.fill(bg_color)

            #run generation for that frame
            for y in range(self.map_size):
                for x in range(self.map_size):
                    self.grid[y][x].draw(self.SCREEN)

            #check if fire has gone out

            #check if program has stopped running
            events = pg.event.get()
            for event in events:
                if event.type == pg.QUIT:
                    RUNNING = False

            #update display
            pg.display.update()
            self.cur_frame += 1

sim1 = Simulation()
sim1.run_sim()
    
