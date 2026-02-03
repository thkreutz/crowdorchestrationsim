from ..constants import rotation_max, acceleration_max
from ..obstacles import *
import numpy as np



class ReplayAgent:

    def __init__(self, positions, velocities, color=(0,200,20), radius=6, start_frame=0): #, fps=30):

        
        # initial position
        self.y = positions[0][0]
        self.x = positions[0][1]
        # all positions
        self.pos_y = positions[0,:]
        self.pos_x = positions[1,:]

        # initial velocity (0)
        self.vel_y = velocities[0][0]
        self.vel_x = velocities[0][1]
        self.vels_y = velocities[0,:]
        self.vels_x = velocities[1,:]
        
        #print(start_frame)

        # Start frame to control when the agent is starting to get replayed
        self.start_frame = start_frame

        # initial heading
        self.ang = np.arctan2(self.vel_y, self.vel_x)

        # set starting position of the agent at tick=0, so next tick = 1
        self.tick = 1
        self.clock_tick_rate = 0.6
        self.clock = self.clock_tick_rate  ### internal clock for synchronization with other replay agents..
       
        # The agent's goal position.
        self.goal = positions[-1]

        # color for visualization
        self.color = color

        # For collision checking
        # alive state -> agent is alive as long as it does not collide with anything
        # circle object for collision detection
        self.alive = False
        self.circle = Circle(self.x, self.y, radius=radius, semantic_class=10)

        # Observation LiDAR
        self.lidar_num_rays = 128
        self.lidar_fov = 2*np.pi
        self.lidar_max_dist = 300
        self.lidar_alias = True # Lidar bins alias into each other
        self.lidar_exp_gain = 1.0
        self.lidar_type = 'pseudo' ## not used, but just for querying.

        # Observations
        self.last_observations = []

        # Set an agent id to put it on the occupancy grid
        self.id = Obstacle.uid
        Obstacle.uid += 1

            
            
    def kill(self):
        # agent collided => kill it
        self.alive = False
        self.color = (255, 0, 0)
    
    def policy_step(self, obs=None):

        # if no policy exists, simply exectute random actions
        vel = np.random.uniform(0, 2) - 1
        ang = np.random.uniform(0, 2) - 1
        return [vel, ang]

        #return self.policy(self, obs)

    def dynamics_step(self):
        
        # If alive -> Start frame reached -> Start replaying
        if self.alive and (self.tick < len(self.pos_x)):
            self.x = self.pos_x[self.tick]
            self.y = self.pos_y[self.tick]
            self.vel_x = self.vels_x[self.tick]
            self.vel_y = self.vels_y[self.tick]
            self.angle = np.arctan2(self.vel_y, self.vel_x)
            self.tick += 1
            self.clock += self.clock_tick_rate
            if self.tick == len(self.pos_x):
                #self.tick = 0
                self.alive = False ## agent is finished

            # move the circle.
            self.circle.x = self.x 
            self.circle.y = self.y

        else:
            # agent is dead and can not move anymore.
            self.x = self.x
            self.y = self.y
            #self.tick += 1
            self.clock += self.clock_tick_rate

    def ego_xy(self, pos):
        # transform global xy to local agent pos
        # based on
        # https://gamedev.stackexchange.com/questions/79765/how-do-i-convert-from-the-global-coordinate-space-to-a-local-space
        relative_x = pos[0] - self.x
        relative_y = pos[1] - self.y
        rotated_x = (np.cos(-self.ang) * relative_x) - (np.sin(-self.ang) * relative_y)
        rotated_y = (np.cos(-self.ang) * relative_y) + (np.sin(-self.ang) * relative_x)
        return np.array([rotated_x, rotated_y])

    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    def cast_rays(self, win, occcupancy_grid):
        # field of view -> 2pi = 360 degree
        angles = np.linspace(0, self.lidar_fov, self.lidar_num_rays)

        # store the point where an obstacle was hit for each ray
        obs = []
        # for each ray
        for angle in angles:
            # maximum depth
            for depth in range(self.lidar_max_dist):
                target_x = self.x - math.sin(angle) * depth
                target_y = self.y + math.cos(angle) * depth
                if target_x >= occcupancy_grid.shape[0]-1 or target_x < 0:
                    break
                if target_y >= occcupancy_grid.shape[1]-1 or target_y < 0:
                    break

                col = int(target_x)
                row = int(target_y)

                if (occcupancy_grid[col][row] > 0) and (occcupancy_grid[col][row] != self.id):
                    #print("hit")
                    obs.append((target_x, target_y))
                    break
                    #pygame.draw.line(win, (255, 255, 0), (self.x, self.y), (target_x, target_y), 2)
        return obs