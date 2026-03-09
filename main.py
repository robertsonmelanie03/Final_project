import pygame
import random

COLOR_BACKGROUND = (255, 255, 255)
COLOR_GRID_LINE = (50, 50, 50)
COLOR_FIRE = (255, 80, 0)
#COLOR_LAND = (34, 139, 34)
#COLOR_WATER = (65, 105, 225)
# those are not used in this demo as I am still trying to find a way to represent those
# the color channel I used is to represent foliage

class Block:
    def __init__(self, elevation, foliage_density, wetness, humidity, wind_speed, precipitation, on_fire=False):
        self.elevation = elevation
        self.foliage_density = foliage_density
        self.wetness = wetness
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.precipitation = precipitation
        self.on_fire = on_fire
        self.fire_steps = 0

    def get_flammability(self):
        # calculate score for being on fire
        fuel = self.foliage_density
        moisture_factor = (1 - self.wetness) * (1 - self.humidity) * (1 - self.precipitation / 10)
        wind_factor = self.wind_speed / 20
        score = fuel * moisture_factor * (1 + wind_factor)
        return score

    def try_ignite(self, neighbor_flammability=None):
        # check if neighbor is on fire
        base = self.get_flammability()
        # if so, add some probability
        if neighbor_flammability:
            prob = base + 0.3 * neighbor_flammability
        else:
            prob = base
        return random.random() < prob

    def get_color(self):
        # return color using different block attributes
        if self.on_fire:
            # red if on fire
            return COLOR_FIRE
        else:
            # the deeper the greeness if more foliage
            intensity = int(255 * (1 - self.foliage_density * 0.5))
            return (34, intensity, 34)


class WeatherModel:
    '''
    this part should be using a weather model to predict attributes for blocks
    but I did not train any so using random attributes for blocks
    '''
    def __init__(self, volatility=0.05):
        self.volatility = volatility

    def predict(self, grid):
        new_weather = []
        for row in grid:
            new_row = []
            for block in row:
                # generate random datasets
                new_humidity = block.humidity + random.uniform(-self.volatility, self.volatility)
                new_wind = block.wind_speed + random.uniform(-2, 2)
                new_precip = block.precipitation + random.uniform(-1, 1)
                new_row.append((new_humidity, new_wind, new_precip))
            new_weather.append(new_row)
        return new_weather


class Simulation:
    def __init__(self, width, height, cell_size=30, seed=42):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = []
        self.weather_model = WeatherModel(volatility=0.05)

        # initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Wildfire Spread Simulation (Demo)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        random.seed(seed)
        self._init_terrain_and_weather()
        self._init_fire()

        self.step_count = 0
        self.running = True
        self.max_fire_steps = 5

    def _init_terrain_and_weather(self):
        # generate terrain and weather attributes at random
        
        #Optimization Elio 
        self.grid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                elevation = random.uniform(0, 100)
                foliage = random.uniform(0, 1)
                wetness = random.uniform(0, 1)
                humidity = random.uniform(0, 1)
                wind = random.uniform(0, 10)
                precip = random.uniform(0, 5)
                block = Block(elevation, foliage, wetness,
                              humidity, wind, precip, on_fire=False)
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
        # use the weather model to predict weather attributes
        new_weather = self.weather_model.predict(self.grid)
        for i in range(self.height):
            for j in range(self.width):
                h, w, p = new_weather[i][j]
                self.grid[i][j].humidity = h
                self.grid[i][j].wind_speed = w
                self.grid[i][j].precipitation = p

        # use another 2D array to store updated blocks
        new_fire = [[b.on_fire for b in row] for row in self.grid]

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j].on_fire:
                    # if the block is on fire, try to ignite its neighbors #, add the diagonal fire spread
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            neighbor = self.grid[ni][nj]
                            if not neighbor.on_fire:
                                # check if that neighbor is ignited based on calculated score
                                if neighbor.try_ignite(self.grid[i][j].get_flammability()):
                                    new_fire[ni][nj] = True

        # update on block array
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j].on_fire = new_fire[i][j]

        self.step_count += 1

        #this works
        for i in range(self.height):
            for j in range(self.width):
                 if self.grid[i][j].on_fire:
                    self.grid[i][j].fire_steps += 1
                    if self.grid[i][j].fire_steps >= self.max_fire_steps:
                        self.grid[i][j].on_fire = False
                        self.grid[i][j].fire_steps = 0

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

    def run(self, fps):
        # call all functions
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(fps)

        pygame.quit()

def main():
    sim = Simulation(width=10, height=10, cell_size=40, seed=42)
    sim.run(fps=2)

if __name__ == "__main__":
    main()
