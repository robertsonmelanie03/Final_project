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
    def __init__(self, t, pos, SCREEN):
        self.pos = pos
        self.elevation = t
        coin = random.random()
                # Basic color mapping
        self.SCREEN = SCREEN

        if t >= 10000:
            self.color = (255, 255, 255)
        elif t >= 9000:
            self.color = (123, 103, 103)
        elif t >= 8000:
            self.color = (22, 105, 41)
        elif t >= 6900:
            self.color = (136, 163, 26)
        elif t >= 5000:
            self.color = (186, 189, 30)
        elif t >= 4500:
            self.color = (202, 212, 59)
        elif t >= 1900:
            self.color = (237, 225, 90)
        else:
            self.color = (237, 225, 90)
        
        self.count = 0
       
    def draw(self, SCREEN):
        with pg.PixelArray(SCREEN) as pixel_array:
            pixel_array[self.pos[0]][self.pos[1]] = self.color

    def erode(self, map):

        hood = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        for d in hood:
            p = [self.pos[0] + d[0], self.pos[1] + d[1]]
            if p[0] < 0 or p[0] >= len(map) or p[1] < 0 or p[1] >= len(map):
                pass
            else:
                new_elevation = map[p[0]][p[1]].elevation - 50
                map[p[0]][p[1]] = Block(new_elevation, p, self.SCREEN)
                for dir in hood:
                    p = [self.pos[0] + dir[0], self.pos[1] + dir[1]]
                    if p[0] < 0 or p[0] >= len(map) or p[1] < 0 or p[1] >= len(map):
                        pass
                    else:
                        new_elevation = map[p[0]][p[1]].elevation - 35
                        map[p[0]][p[1]] = Block(new_elevation, p, self.SCREEN)

        

    def river(self):
        self.color = (0, 0, 255)

    def lake(self):
        x = self.pos[0]
        y = self.pos[1]
        rect = (x, y, random.randint(5, 20), random.randint(5, 20))
        color = (0, 0, 255)
        pg.draw.ellipse(self.SCREEN, color, rect)

def generate_erosion(map, seed, num_events, N, SCREEN):
    random.seed(seed)
    mapsize = 2**N
    #initialize neighborhood
    neighborhood = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    for _ in range(num_events):
        startpoint = [random.randint(1, mapsize - 2), random.randint(1, mapsize - 2)]
        p = startpoint
        max_steps = 40
        steps = 0
        check = False
        while check == False and steps <= max_steps:
            probs = []
            valid_neighbors = []
            # Calculate gradient for each neighbor
            for i, d in enumerate(neighborhood):
                point = [p[0] + d[0], p[1] + d[1]]
                # Bounds check
                if point[0] < 0 or point[0] >= len(map) or point[1] < 0 or point[1] >= len(map):
                    probs.append(0)
                    continue
                grade = map[point[0]][point[1]].elevation - map[p[0]][p[1]].elevation
                if grade < 0:
                    # Probability based on steepness (more steep = more likely)
                    prob = 2.71828**(-0.05 * grade)  # Removed the negative sign
                    probs.append(prob)
                    valid_neighbors.append((i, point, grade))
                else:
                    probs.append(0)
            
            # Normalize probabilities
            total_prob = sum(probs)
            if total_prob == 0:
                check = True
                break
            
            newprobs = [p / total_prob for p in probs]
            
            # Select neighbor based on probability
            r = random.random()
            cumulative = 0
            moved = False
            for i, prob in enumerate(newprobs):
                cumulative += prob
                if r <= cumulative:
                    #move to neighbor
                    d = neighborhood[i]
                    point = [p[0] + d[0], p[1] + d[1]]
                    
                    #erode neighbor
                    current_elevation = map[point[0]][point[1]].elevation
                    new_elevation =  current_elevation - 40
                    map[point[0]][point[1]] = Block(new_elevation, point, SCREEN)
                    
                    #erode current point 
                    current_p_elevation = map[p[0]][p[1]].elevation
                    map[p[0]][p[1]] = Block(current_p_elevation - 20, p, SCREEN)
                    
                    p = point
                    moved = True
                    break
            
            if not moved:
                #just erode current point
                current_elevation = map[p[0]][p[1]].elevation
                new_elevation = current_elevation - 20
                map[p[0]][p[1]] = Block(new_elevation, p, SCREEN)
                check = True
            
            steps += 1


def midpoint_dis():
    pass

##MIDPOINT DISPLACE ALGORITHIM CITATION: https://gamedev.stackexchange.com/questions/72180/midpoint-displacement-1d-how-to-properly-reduce-my-random-number-range
# I USE THIS FOR RIVER GENERATION
import random
import matplotlib.pyplot as plt

def displace(points, max_disp, ):
    """
    Performs a single iteration of midpoint displacement on a list of points.
    
    The new midpoint's y-coordinate is displaced by a random value 
    within the range [-max_disp, +max_disp].
    """
    new_points = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]

        # Calculate midpoint coordinates
        midpoint_x = 0.5 * (p1[0] + p2[0])
        midpoint_y = 0.5 * (p1[1] + p2[1])

        # Displace the y-coordinate by a random amount
        midpoint_y += random.uniform(-max_disp, +max_disp)
        
        new_points.append(p1)
        new_points.append((midpoint_x, midpoint_y))
    
    new_points.append(points[-1]) # Add the last endpoint
    return new_points

def generate_fractal_line(iterations, initial_disp, points, scale_factor=0.8):
    """
    Generates a fractal line using the midpoint displacement algorithm.
    
    Args:
        iterations: The number of recursive iterations to run.
        initial_disp: The initial maximum displacement value.
        scale_factor: The factor by which max_disp is reduced each iteration.
        
    Returns:
        A list of (x, y) tuples representing the fractal line.
    """
    # Start with initial endpoints (0, 0) and (1, 0) - can be customized
   
    max_disp = initial_disp

    for _ in range(iterations):
        points = displace(points, max_disp)
        max_disp *= scale_factor # Reduce displacement for the next iteration
        
    return points


def generate_rivers(map, seedpoint, N, SCREEN):

    num = 4
    chunk = int(2**N / num)
    hood = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1], [-2, -1], [-2, 1], [2, 0]]

    for x in range(num):
        for y in range(num):
            riverpoints = []
            chunk_ar = np.empty((chunk, chunk))
            #iterate over each chunk to find river path
            for r in range(x * chunk, x * chunk + chunk):
                row_list = []
                for c in range(y * chunk, y * chunk + chunk):
                    chunk_ar[r - x * chunk][c - y * chunk] = map[r][c].elevation
            #find max and min points
            max_index = np.argmax(chunk_ar)
            max_index = np.unravel_index(max_index, chunk_ar.shape)
            min_index = np.argmin(chunk_ar)
            min_index = np.unravel_index(min_index, chunk_ar.shape)

            p = [min_index, max_index]
            #generate riverpoints using midpoint displacement algo
            riverpoints = generate_fractal_line(iterations = 9, initial_disp = 40, points = p, scale_factor=0.5)
            
            #draw each river
            for p in riverpoints:
                p = list(p)
                p[0] = round(p[0]) + x*chunk
                p[1] = round(p[1]) + y*chunk
                if p[0] < 0 or p[0] >= len(map) or p[1] < 0 or p[1] >= len(map):
                    pass
                else:
                    map[p[0]][p[1]].erode(map)
            for p in riverpoints:
                p = list(p)
                p[0] = round(p[0]) + x*chunk
                p[1] = round(p[1]) + y*chunk
                if p[0] < 0 or p[0] >= len(map) or p[1] < 0 or p[1] >= len(map):
                    pass
                else:
                    map[p[0]][p[1]].river()
            endpoint = [riverpoints[0][0] + x*chunk, riverpoints[0][1] + y*chunk] 
            for r in range(chunk):
                for c in range(chunk):
                    if map[r + x*chunk][c + y*chunk].elevation < map[endpoint[0]][endpoint[1]].elevation + 1000 and map[r + x*chunk][c + y*chunk].elevation > map[endpoint[0]][endpoint[1]].elevation - 1000:
                        map[r + x*chunk][c + y*chunk].river()
            map[endpoint[0]][endpoint[1]].lake()


def generate(N, seed_, SCREEN):
    map1 = DS.diamond_square(shape=(2**N + 1, 2**N + 1), 
                         min_height=1, 
                         max_height=16000,
                         roughness=0.74, random_seed=seed_)

    terrain = [[0 for _ in range(2**N + 1)] for _ in range(2**N + 1)]

    for x in range(2**N + 1):
        for y in range(2**N + 1):
            val = map1[x][y]
            pos = [x, y]
            terrain[x][y] = Block(val, pos, SCREEN)
    
    pos = [0, 0]
    
    
    generate_erosion(terrain, seed_, 5000, N, SCREEN)
    generate_rivers(terrain, pos, N, SCREEN)
    generate_erosion(terrain, seed_, 5000, N, SCREEN)

    
    #get prep count for each pixel to generate river
    
    #draw rivers
    #get_rivers(terrain, N)


    return terrain


class Simulation:
    def __init__(self):
        N = 9
        self.map_size = 2**N + 1
        self.SCREEN = pg.display.set_mode((2**N + 1, 2**N + 1))
        seed = 8990333
        #define grid of pixels
        self.grid = generate(N, seed, self.SCREEN)

        
    

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
    
    
        
