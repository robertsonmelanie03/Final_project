import pygame
import random
import pandas as pd
from sklearn import metrics, model_selection, tree
from sklearn.ensemble import RandomForestRegressor
import pdb

data = pd.read_csv('datasets/forestfires_preprocessed.csv')

COLOR_BACKGROUND = (255, 255, 255)
COLOR_GRID_LINE = (50, 50, 50)
COLOR_FIRE = (255, 80, 0)

class Block:
    # the Block class stores weather information and has methods about spreading fire
    def __init__(self, temp, rh, wind, rain, on_fire=False):
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

    def get_flammability(self):
        # calculate a probability of being ignited
        # those factors are striped based on orginal data range
        temp_factor = max(0, min(1, (self.temp - 10) / 30))
        rh_factor = max(0, (1 - self.rh / 100.0))
        wind_factor = min(self.wind / 20.0, 1.0)
        rain_factor = max(0, 1 - self.rain / 5.0)
        # use a product to simulate as if any factor is very low, than the product should be as well
        flammability = temp_factor * rh_factor * wind_factor * rain_factor
        # scale the flammability so we can see the change more quickly
        # but no larger than 1.0 as it is a probability
        return min(1.0, flammability * 7)

    def try_ignite(self, neighbor_flammability=None):
        # return True if ignition succeed
        base = self.get_flammability()
        # calculate the probability ignite neighbors
        prob = base + (0.3 * neighbor_flammability if neighbor_flammability else 0)
        return random.random() < min(prob, 1.0)

    def get_color(self):
        # return a color that represent danger of being ignited
        # the brighter the color in g, the more likely to catch on fire
        # handle the condition of already on fire independently
        if self.on_fire:
            return COLOR_FIRE
        f = self.get_flammability()
        r = int(f * 255)
        g = int((1 - f) * 255)
        b = 0
        return (r, g, b)

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
    def __init__(self, width, height, cell_size=30, max_fire_steps=12, seed=42):
        # set up boundry of the pygame window, and grid size
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = []
        self.models = WeatherModel()
        # initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
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

    def _init_terrain_and_weather(self):
        # generate terrain and weather attributes at random
        #Optimization Elio 
        self.grid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                temp = random.randint(15,30) + random.random()
                rh = random.randint(20,50) + random.random()
                wind = random.randint(0,20) + random.random()
                rain = random.randint(0,5) + random.random()
                block = Block(temp, rh, wind, rain)
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
                # use the model to predict and store prediction in the block as current info
                self.grid[i][j].temp = min(40.0, max(0.0, self.models.temp_model.predict([[self.grid[i][j].prev_temp, self.grid[i][j].prev_rh, self.grid[i][j].prev_wind, self.grid[i][j].prev_rain, self.grid[i][j].prev_avg_neighbor_temp, self.grid[i][j].prev_avg_neighbor_rh, self.grid[i][j].prev_avg_neighbor_wind, self.grid[i][j].prev_avg_neighbor_rain]])[0]))
                self.grid[i][j].rh = min(100.0, max(0.0, self.models.rh_model.predict([[self.grid[i][j].prev_temp, self.grid[i][j].prev_rh, self.grid[i][j].prev_wind, self.grid[i][j].prev_rain, self.grid[i][j].prev_avg_neighbor_temp, self.grid[i][j].prev_avg_neighbor_rh, self.grid[i][j].prev_avg_neighbor_wind, self.grid[i][j].prev_avg_neighbor_rain]])[0]))
                self.grid[i][j].wind = min(20.0, max(0.0, self.models.wind_model.predict([[self.grid[i][j].prev_temp, self.grid[i][j].prev_rh, self.grid[i][j].prev_wind, self.grid[i][j].prev_rain, self.grid[i][j].prev_avg_neighbor_temp, self.grid[i][j].prev_avg_neighbor_rh, self.grid[i][j].prev_avg_neighbor_wind, self.grid[i][j].prev_avg_neighbor_rain]])[0]))
                self.grid[i][j].rain = min(10.0, max(0.0, self.models.rain_model.predict([[self.grid[i][j].prev_temp, self.grid[i][j].prev_rh, self.grid[i][j].prev_wind, self.grid[i][j].prev_rain, self.grid[i][j].prev_avg_neighbor_temp, self.grid[i][j].prev_avg_neighbor_rh, self.grid[i][j].prev_avg_neighbor_wind, self.grid[i][j].prev_avg_neighbor_rain]])[0]))
        # try to ignite neighbors
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
                    # fire goes out after certain number of frames
                    if self.grid[i][j].fire_steps >= self.max_fire_steps:
                        self.grid[i][j].on_fire = False
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

    def run(self, fps):
        # call all functions
        while self.running:
            self.update()
            self.handle_events()
            self.draw()
            self.clock.tick(fps)
            # debug
            #print(f"Step {self.step_count}: temp at (0,0) = {self.grid[0][0].rain}, temp at (0,1) = {self.grid[0][1].rain}")
        pygame.quit()

def main():
    sim = Simulation(width=10, height=10, cell_size=40, max_fire_steps=12, seed=42)
    sim.run(fps=0.5)

if __name__ == "__main__":
    main()
