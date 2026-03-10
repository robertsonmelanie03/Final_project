import pygame
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from hkb_diamondsquare import DiamondSquare as DS
import pdb

data = pd.read_csv('datasets/forestfires_preprocessed.csv')

COLOR_BACKGROUND = (255, 255, 255)
COLOR_GRID_LINE = (50, 50, 50)
COLOR_FIRE = (255, 80, 0)

def generate_erosion(elevation, num_events=5000, seed=42):
    # initialize neighborhood
    random.seed(seed)
    mapsize = elevation.shape[0]
    neighborhood = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    
    for _ in range(num_events):
        startpoint = [random.randint(1, mapsize - 2), random.randint(1, mapsize - 2)]
        p = startpoint
        max_steps = 40
        steps = 0
        check = False
        while not check and steps <= max_steps:
            probs = []
            valid_neighbors = []
            # calculate gradient for each neighbor
            for i, d in enumerate(neighborhood):
                ni, nj = p[0] + d[0], p[1] + d[1]
                # Bounds check
                if ni < 0 or ni >= mapsize or nj < 0 or nj >= mapsize:
                    probs.append(0)
                    continue
                grade = elevation[ni, nj] - elevation[p[0], p[1]]
                if grade < 0:
                    # probability based on steepness (more steep = more likely)
                    # removed the negative sign
                    prob = 2.71828 ** (-0.05 * grade)
                    probs.append(prob)
                    valid_neighbors.append((i, (ni, nj), grade))
                else:
                    probs.append(0)
            
            total = sum(probs)
            if total == 0:
                check = True
                break
            
            # normalize probabilities
            probs = [p / total for p in probs]
            
            # select neighbor based on probability
            r = random.random()
            cum = 0
            moved = False
            for i, prob in enumerate(probs):
                cum += prob
                if r <= cum:
                    # move to neighbor
                    d = neighborhood[i]
                    ni, nj = p[0] + d[0], p[1] + d[1]
                    # erode neighbor
                    elevation[ni, nj] -= 40
                    # erode current point
                    elevation[p[0], p[1]] -= 20
                    p = [ni, nj]
                    moved = True
                    break
            
            if not moved:
                # just erode current point
                elevation[p[0], p[1]] -= 20
                check = True
            steps += 1
    # return a mapping of erosion to the position of blocks
    return elevation

def generate_rivers(elevation, N, seed):
    # MIDPOINT DISPLACE ALGORITHIM CITATION: https://gamedev.stackexchange.com/questions/72180/midpoint-displacement-1d-how-to-properly-reduce-my-random-number-range
    def displace(points, max_disp):
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
            max_disp *= scale_factor
        return points
    
    random.seed(seed)
    # this is the numpy format of return of original generate_erosion()
    mapsize = elevation.shape[0]
    num = 4
    chunk = int(mapsize // num)
    # similarly, a mask is used to mapping is_river/lake or not to blocks' positions
    river_mask = np.zeros_like(elevation, dtype=bool)
    lake_mask = np.zeros_like(elevation, dtype=bool)

    for x in range(num):
        for y in range(num):
            # iterate over each chunk to find river path
            x0, y0 = x * chunk, y * chunk
            x1 = min(x0 + chunk, mapsize)
            y1 = min(y0 + chunk, mapsize)
            block = elevation[x0:x1, y0:y1]
            if block.size == 0:
                continue
            # find max and min points
            max_idx = np.unravel_index(np.argmax(block), block.shape)
            min_idx = np.unravel_index(np.argmin(block), block.shape)

            start = (x0 + min_idx[0], y0 + min_idx[1])
            end = (x0 + max_idx[0], y0 + max_idx[1])
            # generate riverpoints using midpoint displacement algo
            points = generate_fractal_line(iterations=9, initial_disp=40, points=[start, end], scale_factor=0.5)
            
            # draw each river
            for p in points:
                px, py = int(round(p[0])), int(round(p[1]))
                if 0 <= px < mapsize and 0 <= py < mapsize:
                    elevation[px, py] -= 40
                    river_mask[px, py] = True
            end_x, end_y = int(round(points[-1][0])), int(round(points[-1][1]))
            # draw lakes at the end of rivers
            lake_radius_x = random.randint(5, 20)
            lake_radius_y = random.randint(5, 20)
            for dx in range(-lake_radius_x, lake_radius_x+1):
                for dy in range(-lake_radius_y, lake_radius_y+1):
                    if (dx/lake_radius_x)**2 + (dy/lake_radius_y)**2 <= 1:
                        nx, ny = end_x + dx, end_y + dy
                        if 0 <= nx < mapsize and 0 <= ny < mapsize:
                            lake_mask[nx, ny] = True
                            elevation[nx, ny] -= 100
    # return a mapping of rivers and lakes to positions of blocks
    return river_mask, lake_mask

class Block:
    # the Block class stores weather information and has methods about spreading fire
    def __init__(self, temp, rh, wind, rain, elevation, on_fire=False, is_river=False, is_lake=False):
        # store terrain features
        self.elevation = elevation
        self.is_river = is_river
        self.is_lake = is_lake
        # store current weather attributes
        self.temp = temp
        self.rh = rh
        self.wind = wind
        self.rain = rain
        # store previous weather attributes
        self.prev_temp = 0.0
        self.prev_rh = 0.0
        self.prev_wind = 0.0
        self.prev_rain = 0.0
        # store previous weather attributes of its neighbors average
        self.prev_avg_neighbor_temp = 0.0
        self.prev_avg_neighbor_rh = 0.0
        self.prev_avg_neighbor_wind = 0.0
        self.prev_avg_neighbor_rain = 0.0
        # fire status, fire extinquishes after certain steps defined in Simulation class
        self.on_fire = on_fire
        self.fire_steps = 0
        self.burned = True

    def get_flammability(self):
        # calculate a probability of being ignited
        # those factors are striped based on orginal data range
        temp_factor = max(0, min(1, (self.temp - 10) / 30))
        rh_factor = max(0, (1 - self.rh / 100.0))
        wind_factor = min(self.wind / 20.0, 1.0)
        rain_factor = max(0, 1 - self.rain / 5.0)
        # if it is too high or it is a river or lake, not flammable
        if self.elevation >= 9000 or self.is_river or self.is_lake:
            return 0.0
        # otherwise the lower the more likely to catch on fire
        elif self.elevation >= 7000:
            elevation_factor = max(0, 1 - (self.elevation - 7000) / 2000)
        else:
            elevation_factor = 1.0
        # use a product to simulate as if any factor is very low, than the product should be as well
        flammability = temp_factor * rh_factor * wind_factor * rain_factor * elevation_factor
        # scale the flammability so we can see the change more quickly
        # but no larger than 1.0 as it is a probability
        return min(1.0, flammability * 7)

    def try_ignite(self, neighbor_flammability=None):
        if self.rain >= 5:
            return False
        # return True if ignition succeed
        base = self.get_flammability()
        # calculate the probability ignite neighbors
        prob = base + (0.3 * neighbor_flammability if neighbor_flammability else 0)
        return random.random() < min(prob, 1.0)

    def get_color(self):
        # return a color that represent elevation information
        # if it is a river or lake, show that at first priority
        if self.is_river or self.is_lake:
            return (0, 0, 255)
        # fire or not as second
        if self.on_fire:
            return COLOR_FIRE
        elif self.burned:
            return (50, 50, 50)
        # get color for certain elevation range
        if self.elevation >= 10000:
            return (255, 255, 255)
        elif self.elevation >= 9000:
            return (123, 103, 103)
        elif self.elevation >= 8000:
            return (22, 105, 41)
        elif self.elevation >= 6900:
            return (136, 163, 26)
        elif self.elevation >= 5000:
            return (186, 189, 30)
        elif self.elevation >= 4500:
            return (202, 212, 59)
        elif self.elevation >= 1900:
            return (237, 225, 90)
        else:
            return (237, 225, 90)

class WeatherModel:
    # the WeatherModel class store different models as attributes
    # and be instantiated in Simulation.run() method so that we can use those models
    def __init__(self):
        self.temp_model = self.train_model(['prev_temp', 'prev_RH', 'prev_wind', 'prev_rain', 'prev_avg_neighbor_temp', 'prev_avg_neighbor_RH', 'prev_avg_neighbor_wind', 'prev_avg_neighbor_rain'], 'target_temp')
        self.rh_model = self.train_model(['prev_temp', 'prev_RH', 'prev_wind', 'prev_rain', 'prev_avg_neighbor_temp', 'prev_avg_neighbor_RH', 'prev_avg_neighbor_wind', 'prev_avg_neighbor_rain'], 'target_RH')
        self.wind_model = self.train_model(['prev_temp', 'prev_RH', 'prev_wind', 'prev_rain', 'prev_avg_neighbor_temp', 'prev_avg_neighbor_RH', 'prev_avg_neighbor_wind', 'prev_avg_neighbor_rain'], 'target_wind')
        self.rain_model = self.train_model(['prev_temp', 'prev_RH', 'prev_wind', 'prev_rain', 'prev_avg_neighbor_temp', 'prev_avg_neighbor_RH', 'prev_avg_neighbor_wind', 'prev_avg_neighbor_rain'], 'target_rain')

    def train_model(self, input_columns=[], target_col=''):
        # pending
        X = data[input_columns].values
        Y = data[target_col].values
        model = RandomForestRegressor()
        model.fit(X, Y)
        #print('training score =', model.score(train_X, train_Y))
        #print('testing score =', model.score(test_X, test_Y ))
        return model

class Simulation:
    # the Simulation class brings up everyting together and has a run() method calls everything
    def __init__(self, N=8, cell_size=4, max_fire_steps=12, seed=42):
        self.grid_size = 2**N + 1
        self.width = self.grid_size
        self.height = self.grid_size
        self.cell_size = cell_size
        self.seed = seed
        self.grid = []
        self.models = WeatherModel()
        # generate terrain and simulate erosion
        self.elevation_grid = self._generate_terrain(N, seed)
        self.elevation_grid = generate_erosion(self.elevation_grid, num_events=5000, seed=seed)
        self.river_mask, self.lake_mask = generate_rivers(self.elevation_grid, N, seed=seed)
        # initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
        pygame.display.set_caption("Wildfire Spread Simulation (Demo)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        # based on a seed, generate terrain, and start a fire
        random.seed(seed)
        self._init_terrain_and_weather()
        self._init_fire()
        # count how many frames being passed and store ending conditions
        self.step_count = 0
        self.running = True
        self.max_fire_steps = max_fire_steps

    def _generate_terrain(self, N, seed):
        # generate a map of elevation
        elevation = DS.diamond_square(
        shape=(2**N + 1, 2**N + 1),
        min_height=1,
        max_height=16000,
        roughness=0.74,
        random_seed=seed)
        return elevation

    def _init_terrain_and_weather(self):
        self.grid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                # initializa blocks' attributes based on given maps
                e = self.elevation_grid[i][j]
                is_river = self.river_mask[i][j]
                is_lake = self.lake_mask[i][j]
                # generate random weather attributes based on elevation
                temp = 30 - (e / 16000) * 20
                rh = 50 + (e / 16000) * 30
                wind = 5 + (e / 16000) * 15
                rain = 2 + (e / 16000) * 3
                temp += random.uniform(-2, 2)
                rh += random.uniform(-5, 5)
                wind += random.uniform(-2, 2)
                rain += random.uniform(-1, 1)
                temp = max(0, min(50, temp))
                rh = max(0, min(100, rh))
                wind = max(0, min(50, wind))
                rain = max(0, min(20, rain))
                # generate a block and append to the grid 2D array
                block = Block(temp=temp, rh=rh, wind=wind, rain=rain, elevation=e, is_river=is_river, is_lake=is_lake)
                row.append(block)
            self.grid.append(row)

    def _init_fire(self):
        # ignite a block to start spreading fire (center for example)
        ci, cj = self.height // 2, self.width // 2
        self.grid[ci][cj].on_fire = True

    def reset(self):
        # reset everything to a fixed seed
        random.seed(42) 
        self._init_terrain_and_weather()
        self._init_fire()
        self.step_count = 0

    def update(self):
        # use another 2D array to store updated blocks
        new_fire = [[b.on_fire for b in row] for row in self.grid]
        # update block information
        # update previous information
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j].prev_temp = self.grid[i][j].temp
                self.grid[i][j].prev_rh = self.grid[i][j].rh
                self.grid[i][j].prev_wind = self.grid[i][j].wind
                self.grid[i][j].prev_rain = self.grid[i][j].rain
        # update neighbors previous information
        for i in range(self.height):
            for j in range(self.width):
                neighbor_temp_list = []
                neighbor_rh_list = []
                neighbor_wind_list = []
                neighbor_rain_list = []
                # now count blocks in 8 directions as neighbors
                for di, dj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.height and 0 <= nj < self.width:
                        neighbor = self.grid[ni][nj]
                        neighbor_temp_list.append(neighbor.prev_temp)
                        neighbor_rh_list.append(neighbor.prev_rh)
                        neighbor_wind_list.append(neighbor.prev_wind)
                        neighbor_rain_list.append(neighbor.prev_rain)
                # calculate input columns that model needs and store in that block
                self.grid[i][j].prev_avg_neighbor_temp = sum(neighbor_temp_list) / len(neighbor_temp_list)
                self.grid[i][j].prev_avg_neighbor_rh = sum(neighbor_rh_list) / len(neighbor_rh_list)
                self.grid[i][j].prev_avg_neighbor_wind = sum(neighbor_wind_list) / len(neighbor_wind_list)
                self.grid[i][j].prev_avg_neighbor_rain = sum(neighbor_rain_list) / len(neighbor_rain_list)
        # collect all features of all blocks at the frame to numpy arrays
        features = []
        for i in range(self.height):
            for j in range(self.width):
                b = self.grid[i][j]
                features.append([b.prev_temp, b.prev_rh, b.prev_wind, b.prev_rain,
                                b.prev_avg_neighbor_temp, b.prev_avg_neighbor_rh,
                                b.prev_avg_neighbor_wind, b.prev_avg_neighbor_rain])
        # get predicted weather attribute arrays
        temps = self.models.temp_model.predict(features)
        rhs = self.models.rh_model.predict(features)
        winds = self.models.wind_model.predict(features)
        rains = self.models.rain_model.predict(features)
        # update each block and stripe the extreme data
        idx = 0
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j].temp = min(40, max(0, temps[idx]))
                self.grid[i][j].rh = min(100, max(0, rhs[idx]))
                self.grid[i][j].wind = min(20, max(0, winds[idx]))
                self.grid[i][j].rain = min(10, max(0, rains[idx]))
                idx += 1
        # try to ignite neighbor
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j].on_fire:
                    for di, dj in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            neighbor = self.grid[ni][nj]
                            if not neighbor.on_fire:
                                # append the fire status to a 2D array that maps where the fired block is
                                if neighbor.try_ignite(self.grid[i][j].get_flammability()):
                                    new_fire[ni][nj] = True
        # update the fire status for the mapped blocks afterwards
        # because if fire status is changed in the above loop, the newly ignited block may instantly begin to ignite others
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j].on_fire = new_fire[i][j]

        # update information for blocks that is on fire
        for i in range(self.height):
            for j in range(self.width):
                 # maintain a counter for frames being on fire
                 # increment to some weather attributes that can be influenced by fire
                 if self.grid[i][j].on_fire:
                    self.grid[i][j].fire_steps += 1
                    self.grid[i][j].rh = max(0, min(100, self.grid[i][j].rh - 10))
                    self.grid[i][j].temp = max(0, min(50, self.grid[i][j].temp + 2))
                    # fire goes out after certain number of frames or there is a heavy rain
                    if self.grid[i][j].fire_steps >= self.max_fire_steps or self.grid[i][j].rain >= 3:
                        self.grid[i][j].on_fire = False
                        self.burned = True
                        self.grid[i][j].fire_steps = 0
        # count one frame passed
        self.step_count += 1
       
    def draw(self):
        self.screen.fill(COLOR_BACKGROUND)
        # draw grid using color in the Block
        for i in range(self.height):
            for j in range(self.width):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                color = self.grid[i][j].get_color()
                pygame.draw.rect(self.screen, color, rect)
                # draw lines to seperate grid
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)
        pygame.display.flip()

    def handle_events(self):
        # check events, exit if press quit, reset if press r
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                # rain if press 1, increasing precipitation and relative humidity
                elif event.key == pygame.K_1:
                    for row in self.grid:
                        for block in row:
                            block.rain = min(10, block.rain + 1)
                            block.rh = min(100, block.rh + 5)
                            if block.on_fire and block.rain >= 5:
                                block.on_fire = False
                                block.fire_steps = 0
                # temp up if press 2
                elif event.key == pygame.K_2:
                    for row in self.grid:
                        for block in row:
                            block.temp = min(50, block.temp + 2)
                # wind speed up if press 3
                elif event.key == pygame.K_3:
                    for row in self.grid:
                        for block in row:
                            block.wind = min(50, block.wind + 2)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # left click to start a fire in the block
                if event.button == 1:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    col = mouse_x // self.cell_size
                    row = mouse_y // self.cell_size
                    if 0 <= row < self.height and 0 <= col < self.width:
                        self.grid[row][col].on_fire = True
                # right click to put out the fire in the block
                elif event.button == 3:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    col = mouse_x // self.cell_size
                    row = mouse_y // self.cell_size
                    if 0 <= row < self.height and 0 <= col < self.width:
                        self.grid[row][col].on_fire = False
                        self.grid[row][col].fire_steps = 0

    def run(self, fps):
        # call all functions
        while self.running:
            self.update()
            self.handle_events()
            self.draw()
            self.clock.tick(fps)
            # debug
            #pdb.set_trace()
            #print(f"Step {self.step_count}: temp at (0,0) = {self.grid[0][0].rain}, temp at (0,1) = {self.grid[0][1].rain}")
        pygame.quit()

def main():
    sim = Simulation(seed=42)
    sim.run(fps=1)

if __name__ == "__main__":
    main()
