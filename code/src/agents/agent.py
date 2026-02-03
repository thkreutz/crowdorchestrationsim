from src.objects.obstacles import *
import numpy as np
import collections


rotation_max, acceleration_max = 0.08, 0.5

class PedestrianAgent:

    def __init__(self, x=0., y=0., ang=0., vel_x=0., vel_y=0., max_vel=5.0, 
                            goal=(0,0), color=(0,200,20), policy=None, radius=6, 
                            n_history=10, lidar_num_rays=64, max_x=1, max_y=1, aid=1337, spawn_id=1337, destination_id=1337, spawn_frame=0, action_scaler=None): #, fps=30):

        # initial position
        self.x = x
        self.y = y

        self.max_x = float(max_x)
        self.max_y = float(max_y)

        # initial heading
        self.ang = ang
        # initial velocity (0)
        self.vel_x = vel_x
        self.vel_y = vel_y

        # maximum possible velocity
        self.max_vel = max_vel

        # Angle and distance to goal
        self.ang_goal = 0
        self.dist_goal = 1

        # internal policy
        self.policy = None

        # The agent's goal position.
        self.goal = goal
        self.sub_goal = None # Agent should have a subgoal he is currently pursuing... 

        # color for visualization
        self.color = color

        # For collision checking
        # alive state -> agent is alive as long as it does not collide with anything
        # circle object for collision detection
        self.alive = True
        self.circle = Circle(x, y, radius=radius, semantic_class=10)

        # Observation LiDAR
        self.lidar_num_rays = lidar_num_rays
        self.lidar_fov = 2*np.pi
        self.lidar_max_dist = 300
        self.lidar_max_goal_dist = 5 * self.lidar_max_dist  ### distance sensor to the goal is max "larger".
        self.lidar_type = 'pseudo' ## not used, but just for querying.

        self.action_scaler = action_scaler
        # Observations
        ## fill the observation history every tick when we make an observation. maybe every tick is too much @25fps
        ## Should be buffer-like structure
        ## Use queue

        self.observation_history = collections.deque(maxlen=n_history)
        self.n_history = n_history

        # Set an agent id to put it on the occupancy grid
        self.id = aid

        self.dynamics_mode = 'basic'

        self.spawn_id = spawn_id
        self.destination_id = destination_id
        self.finished = False

        self.spawn_frame = spawn_frame
        self.trajectory = [(self.spawn_frame, self.id, self.spawn_id, self.destination_id, self.x, self.y, self.vel_x, self.vel_y)]

    def update_trajectory(self, frame):
        self.trajectory.append((frame, self.id, self.spawn_id, self.destination_id, self.x, self.y, self.vel_x, self.vel_y))

    def set_scaler(self, scaler):
        self.action_scaler = scaler
        self.scaler_set = True

    def kill(self):
        # agent collided => kill it
        self.alive = False
        self.color = (255, 0, 0)
    
    def done(self):
        self.alive = True
        self.finished = True
        self.color = (34,139,34)

    def policy_step(self, obs=None):

        # if no policy exists, simply exectute random actions
        vel = np.random.uniform(0, 2) - 1
        ang = np.random.uniform(0, 2) - 1
        return [vel, ang]

        #return self.policy(self, obs)

    def single_integrator(self, action):

        if self.alive:
            self.x = self.x + action[0] #* dt
            self.y = self.y + action[1] #* dt

            ### Need to update the velocity, which corresponds to the action.
            self.vel_x = action[0]
            self.vel_y = action[1]

            ### Give relative angle and distance to goal as well as a state variable. (im polar coords)
            # relative cartesian 
            x_temp = self.goal[0] - self.x
            y_temp = self.goal[1] - self.y
            self.dist_goal, self.ang_goal = self.cart2pol(x_temp, y_temp)

            # Normalize distance and goal between 0 and 1
            self.dist_goal = min(self.dist_goal, self.lidar_max_goal_dist) / (self.lidar_max_goal_dist)
            self.ang_goal = self.ang_goal / (2 * np.pi)

        else:
            self.x = self.x
            self.y = self.y
            self.vel_x = 0
            self.vel_y = 0


    def unicycle(self, action):
        self.x = self.x + action[0] * np.cos(self.ang) #* dt
        self.y = self.y + action[0] * np.sin(self.ang) #* dt
        self.ang = self.ang + action[1] #* dt

    def dynamics_step(self, action):
        # during replay do inverse scaling because our action space is -1,1

        action = self.action_scaler.inverse_transform([action])[0]

        if self.dynamics_mode == "basic":
            self.single_integrator(action)
        else:
            self.unicycle(action)

    def replay_step(self, action):

         # during replay no scaling
        
        if self.dynamics_mode == "basic":
            self.single_integrator(action)
        else:
            self.unicycle(action)

        #return self.x, self.y
        #return self.x, self.y

    def dynamics_step_legacy(self, action):

        # Crowd simulation by deep RL
        # p_t+1 = p_t + h*v
        # omega_t+1 = omega_t + h*w
        
        if self.alive:
            # action[0]: acceleration | action[1]: rotation
            # ─── APPLY ROTATION ──────────────────────────────────────────────
            self.ang = self.ang + rotation_max * action[1]
            if self.ang > np.pi:
                self.ang = self.ang - 2 * np.pi
            if self.ang < -np.pi:
                self.ang = self.ang + 2 * np.pi
            
            # ─── APPLY ACCELERATION ──────────────────────────────────────────
            acceleration = action[0]
            # backwards acceleration at half thrust
            if acceleration < 0:
                acceleration = acceleration * 0.5

            # upadte velocity and clip to maximum velocity
            self.vel_x = np.clip(self.vel_x + acceleration_max * acceleration * np.cos(self.ang), -self.max_vel, self.max_vel)
            self.vel_y = np.clip(self.vel_y - acceleration_max * acceleration * np.sin(self.ang), -self.max_vel, self.max_vel)
            
            # move agent
            self.x = self.x + self.vel_x
            self.y = self.y + self.vel_y
            self.circle.x = self.x 
            self.circle.y = self.y

            
        else:
            # agent is dead and can not move anymore.
            self.x = self.x
            self.y = self.y

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

    def get_observation(self, occupancy_grid):
        ray_endpoints, dist_obs = self.cast_rays(occupancy_grid)
        internal = self.get_internal_observations()
        # index 0:4 internal, 5: external
        self.observation_history.append(internal + dist_obs)
        return list(self.observation_history), ray_endpoints, internal
        
    def cast_rays(self, occcupancy_grid):
        # field of view -> 2pi = 360 degree
        angles = np.linspace(0, self.lidar_fov, self.lidar_num_rays)

        # store the point where an obstacle was hit for each ray
        obs = []
        dist_obs = []
        # for each ray
        for angle in angles:
            # maximum depth
            for depth in range(self.lidar_max_dist):
                hit = False
                
                target_x = self.x - math.sin(angle) * depth
                target_y = self.y + math.cos(angle) * depth
                if target_x >= occcupancy_grid.shape[0]-1 or target_x < 0:
                    break
                if target_y >= occcupancy_grid.shape[1]-1 or target_y < 0:
                    break

                col = int(target_x)
                row = int(target_y)

                ### Less than 0 means obstacles in  the map, other is the agent id.
                if (occcupancy_grid[col][row] > 0) and (occcupancy_grid[col][row] != self.id):
                    #print("hit")
                    obs.append((target_x, target_y))
                    # compute respective distance
                    hit = True
                    dist_obs.append(np.sqrt((self.x - target_x)**2  +  (self.y - target_y)**2) / self.lidar_max_dist)
                    break
                    # add win as parameter if we still draw line here
                    #pygame.draw.line(win, (255, 255, 0), (self.x, self.y), (target_x, target_y), 2)
            # max distance reached and nothing was found.
            if not hit:
                dist_obs.append(1) ### add LiDAR max dist

        return obs, dist_obs

    def get_internal_observations(self):
        return [self.vel_x, self.vel_y, self.dist_goal, self.ang_goal]
