# -*- coding: utf-8 -*-
#from gym import Env, spaces
import gymnasium as gym
import pygame
import numpy as np
import distinctipy
import json
import cv2
import time

from src.agents.agent import PedestrianAgent

rotation_max, acceleration_max = 0.08, 0.5


def circle_mask(x_cells, y_cells, cx, cy, r):
    mask = (x_cells[np.newaxis,:]- cx)**2 + (y_cells[:,np.newaxis]-cy)**2 < r**2
    return mask


from src.data.scene_loader import TrajScene

class CustomEnv(gym.Env):
    def __init__(self, dataset_name, scene_name, data_root, scale_factor=1, n_agents=1, n_rays=32,
    rendering="human", action_scaler=None, env_config={}):
        #self.observation_space = <gym.space>
        #self.action_space = <gym.space>
        self.action_scaler = action_scaler

        self.dataset_name = dataset_name
        self.scene_name = scene_name
        self.scale_factor = scale_factor
        self.data_root = data_root
        # Read data
        trajscene = TrajScene(dataset_name=dataset_name, scene_name=scene_name, scale_factor=scale_factor, data_root=data_root)
        self.trajs, self.trajs_df, self.coord_transform, self.img_semantic, self.img_reference, self.img_semantic_scaled, self.img_reference_scaled = trajscene.get_scene()
        
        self.steps = 0
        ## specify action and observation space for stable baselines model

        # Env Parameters
        if scale_factor != 1:
            ### IF we use the non_scaled img
            self.img_reference = self.img_reference_scaled
            self.img_semantic = self.img_semantic_scaled
        
        ### Swap Axis of these two because thats how pygame wants it...
        self.img_reference = np.swapaxes(self.img_reference, 0,1 )
        self.img_semantic = np.swapaxes(self.img_semantic, 0,1)

        self.window_width = self.img_reference.shape[1]
        self.window_height = self.img_reference.shape[0]
        self.x_cells = np.arange(self.window_width)
        self.y_cells = np.arange(self.window_height)
        #else:
        #    self.window_width = self.img_reference_scaled.shape[1]
        #    self.window_height = self.img_reference_scaled.shape[0]

        # we define obstacles as elements in the scene that must be avoided + they define the layout of the map.
        self.init_environment_map()

        # Rendering params
        self.rendering = rendering

        self.observations = []

        # "Real" Env Agents
        self.MAX_VELOCITY = 5
        self.N_PAST_TIMESTEPS = 2
        self.N_RAYS = n_rays
        self.n_agents = n_agents
        self.N_internals = 4
        self.agent = self.init_agent()

        self.init_replay_crowd()

        
        self.action_space = gym.spaces.Box(low=np.zeros(2)-1, high=np.ones(2), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.ones(self.N_internals+self.N_RAYS), high=np.ones(self.N_internals+self.N_RAYS), dtype=np.float64)

        

        
    def init_environment_map(self):

        #self.window_width = height
        #self.window_height = width
        
        #### We do a background grid map, where each grid cell has a certain element in its state.
        self.occupancy_grid = np.zeros(self.img_semantic.shape) # exact same shapeas semantic map.

        #print(self.occupancy_grid.shape)
        if self.dataset_name == "eth":
            self.occupancy_grid[np.where(self.img_semantic != 1)] = 1
            
        if self.dataset_name == "sdd":
            self.occupancy_grid[np.where(self.img_semantic == 3)] = 1 #3
            self.occupancy_grid[np.where(self.img_semantic == 5)] = 1 #5

             ### SDD Mapping
            # 1 = road
            # 2 = pavement
            # 3 = structure
            # 4 = terrain
            # 5 = tree

            #if c == 0:
            #    ch[np.where(img_semantic == 1)] = 1
            #    ch[np.where(img_semantic == 2)] = 1
            #    ch[np.where(img_semantic == 4)] = 1
            #if c == 1:
            #    ch[np.where(img_semantic == 3)] = 1
            #    ch[np.where(img_semantic == 5)] = 1
            
        if self.dataset_name == "gc":
            # 0 = walkable space
            # 40 = building
            # 174 = obstacle 
            #[0, 40, 174]
            self.occupancy_grid[np.where(self.img_semantic == 40)] = 1 #3
            self.occupancy_grid[np.where(self.img_semantic == 174)] = 1 #3
        

        if self.dataset_name == "forum":
            # 0 = walkable space
            # 40 = building
            # 174 = obstacle 
            #[0, 40, 174]
            self.occupancy_grid[np.where(self.img_semantic == 113)] = 1 #3
        
        if self.dataset_name == "atc":
            # 0 = walkable space
            # 40 = building
            # 174 = obstacle 
            #[0, 40, 174]
            self.occupancy_grid[np.where(self.img_semantic != 59)] = 1

        #self.occupancy_grid[np.where(self.img_semantic.T) == 2] = 0
        #self.occupancy_grid[np.where(self.img_semantic.T) == 4] = 0 
        
        #self.occupancy_grid[np.where(self.img_semantic.T) == 3] = 1
        #self.occupancy_grid[np.where(self.img_semantic.T) == 5] = 1
        
        #assert(self.occupancy_grid.shape[0] == self.window_width)
        #assert(self.occupancy_grid.shape[1] == self.window_height)
        

        # init all the replay agents on the occupancy grid

    
    def init_replay_crowd(self):
        
        ### init frame
        self.replay_start_frame = self.trajs_df[self.trajs_df.agent_id == self.agent.id].frame_id.values[0]
        self.replay_start_frame = self.replay_start_frame + self.steps
        temp = self.trajs_df[self.trajs_df.frame_id == self.replay_start_frame]
        #print(self.replay_start_frame, len(temp), temp.agent_id.values)

        #print(temp)
        ### Set all other agents present in the same frame on the map.
        for _, row in temp.iterrows():
            if row.agent_id != self.agent.id:
                coords = self.coord_transform.get_pixel_positions(row[["pos_x", "pos_y"]].values) // self.scale_factor
                mask = circle_mask(self.x_cells, self.y_cells, coords[1], coords[0], self.agent.circle.radius)
                self.occupancy_grid[mask] = row.agent_id

    def sample_agent(self, replay_agent=False, replay_pos=None, replay_goal=None, aid=-1):

        if not replay_agent:
            random_choice = list(self.trajs.keys())[np.random.choice(len(self.trajs))]
            traj = self.trajs[random_choice]
            start = self.coord_transform.get_pixel_positions(traj[["start_x", "start_y"]].values[0]) // self.scale_factor
            goal = self.coord_transform.get_pixel_positions(traj[["goal_x", "goal_y"]].values[0]) // self.scale_factor
            return start, goal, random_choice
        else:
            start = self.coord_transform.get_pixel_positions(replay_pos) // self.scale_factor
            goal = self.coord_transform.get_pixel_positions(replay_goal) // self.scale_factor
            return start, goal
    

    def init_agent(self, replay_agent=False, init_specific_agent=False, aid=None):
        # randomly sample x,y positions for each agent within specified range
        #pos_y = np.random.uniform(range_x[0], range_x[1], n_agents)
        #pos_x = np.random.uniform(range_y[0], range_y[1], n_agents)
        #pos = np.column_stack((pos_x, pos_y))

        # use historic start and goal positions from the agents in the dataset to sample start and goal
        cols = np.array(distinctipy.get_colors(5))*255

        if replay_agent or init_specific_agent:
            # mini reset env
            self.agents = []
            self.occupancy_grid = []
            self.observations = []
            self.init_environment_map()

            s = self.trajs[aid][["pos_x", "pos_y"]].values
            pos, goal = self.sample_agent(replay_agent=True, replay_pos=s[0], replay_goal=s[-1])
        else:
            # init  agent
            pos, goal, aid = self.sample_agent(replay_agent=False)


        ag = PedestrianAgent(x=pos[0], y=pos[1], goal=goal, color=(150,20,150), 
                                            radius=3, max_vel=self.MAX_VELOCITY, lidar_num_rays=self.N_RAYS, 
                                            max_x = self.window_width, max_y=self.window_height, aid=aid, action_scaler=self.action_scaler)

        # init the agents on the occupancy grid
        mask = circle_mask(self.x_cells, self.y_cells, ag.y, ag.x, ag.circle.radius)
        self.occupancy_grid[mask] = ag.id

        return ag
    
    def init_render(self):
        # use pygame for rendering
        pygame.init()
        self.window = pygame.display.set_mode((self.window_height, self.window_width) )
        self.clock = pygame.time.Clock()
        # call render once to render all the initial things so 
        # that we can use  them directly to compute the first observation
        self.render()
    
    def reset(self, seed=0, replay_agent=False, aid=None):
        # reset the environment to initial state
        self.agents = []
        self.occupancy_grid = []
        self.observations = []

        self.init_environment_map()
        #self.replay_agent_loader = ETH_Agent_Loader()

        if replay_agent:
            # initialize an agent to be at a specific position.
            # => Can then feed specific actions that are wanted or can be controlled while the crowd plays in the background.
            self.agent = self.init_agent(replay_agent=replay_agent, aid=aid)
        else:
            self.agent = self.init_agent()

        self.steps = 0
        self.init_replay_crowd()

        # reset pygame if rendering = human
        if self.rendering == "human":
            pygame.quit()
            pygame.init()
            self.window = pygame.display.set_mode((self.window_height, self.window_width))
            self.clock = pygame.time.Clock()
            self.render()
        
        

        #return observation
        obs_0, _, _ = self.agent.get_observation(self.occupancy_grid)
        return (np.array(obs_0[-1]).astype(np.float64), {})
    
    def replay_agent(self, actions, aid, render_replay=False):
        
        # reset env
        self.reset(replay_agent=True, aid=aid)

        #print(self.agent.id)
        # we are deterministic, therefore we just need start state and sequence of actions
        states = []
        replay_observations = []

        obs_0, _, _ = self.agent.get_observation(self.occupancy_grid)
        replay_observations.append(obs_0[-1])

        states = [[self.agent.x, self.agent.y]]
        for action in actions:
            if render_replay:
                time.sleep(0.05)
            # perform action
            self.agent.replay_step(action)
            self.steps += 1
            #print(self.steps)

            states.append([self.agent.x, self.agent.y])

            # update grid
            agent_mask = circle_mask(self.x_cells, self.y_cells, self.agent.y, self.agent.x, self.agent.circle.radius)
            
            # clear everything greater than min agentid (10)
            self.occupancy_grid[self.occupancy_grid >= 10] = 0 # remove agent from map, check if new collision collides with env
            #self.occupancy_grid[self.occupancy_grid == self.agent.id] = 0 # remove agent from map, check if new collision collides with env
            
            self.occupancy_grid[agent_mask] = self.agent.id

            self.init_replay_crowd() # draw the frame at agent_start_frame + steps of replay crowd

            # get observation
            obs, ray_points, _ = self.agent.get_observation(self.occupancy_grid)
            replay_observations.append(obs[-1])
            self.observations = ray_points

            if render_replay:
                self.render()
            #pygame.event.pump()

        return np.array(states), np.array(replay_observations)

    def step(self, action):
        ### Perform one step with actions
        #for agent, action in zip(self.agents, actions):
        # should return new observation etc., then we make a list

        # we could have multiple agents... but I just use one.
        self.agent.dynamics_step(action)
        self.steps += 1

        # Check collision.
        agent_mask = circle_mask(self.x_cells, self.y_cells, self.agent.y, self.agent.x, self.agent.circle.radius)
        collision = 0
        self.occupancy_grid[self.occupancy_grid == self.agent.id] = 0 # remove agent from map, check if new collision collides with env
         # remove all agents from the map


        ### COLLISION 
        if np.sum(self.occupancy_grid[np.where(agent_mask == True )]) > 0:
            collision = -1
            done = True
            self.agent.kill()
        
        # Update occupancy grid with new agent's position
        # Clear all replay agents
        self.occupancy_grid[self.occupancy_grid >= 10] = 0
        # Set agent
        self.occupancy_grid[agent_mask] = self.agent.id
        # Set crowd.
        self.init_replay_crowd()

        out_of_frame = 0
        if (self.agent.y >= self.window_width) or (self.agent.x >= self.window_height) or (self.agent.x < 0) or (self.agent.y < 0):
            out_of_frame = -1
            done = True
            self.agent.kill()
        
        ### Compute observations
        self.observations = []
        self.history_observations = []
        self.internal_observations = []
        
        if self.agent.alive:
            # compute agent's observations
            #obs = agent.cast_rays(self.window, self.occupancy_grid)
            #self.observations.append(obs)
            obs, ray_points, internal_only = self.agent.get_observation(self.occupancy_grid)
            obs = np.array(obs)
            self.history_observations = obs
            self.observations = ray_points # ray cast of the agent's last observation
            self.internal_observations = internal_only
        else:
            self.observations = []
            self.history_observations = []
            self.internal_observations = []

        # goal distance reward
        distance_reward = 0
        #distance_reward_weight = 1
        if len(self.history_observations) > 2:
            current_obs = self.history_observations[-1][2]
            previous_obs = self.history_observations[-2][2]
            ## becomes negative if distance to goal becomes greater
            # can include the time of the agent that it takes to reach the goal
            # penalize him the longer the agent takes.
            if previous_obs - current_obs <= 0:

                # Punish for not doing anything or moving away from goal
                distance_reward = -0.00001

        # Punish for taking too long
        time_reward = -0.00001

        
        # speed reward
        #speed_reward = 0
        #if speed <= 1 and speed >= 0.7 :
        #    speed_reward = 0.5 ## reward moving at desired high speed. for reaching the maximum velocity -> can get as close as possible

            # Give reward for a) not colliding, b) moving towards the goal
            

            #observation = obs
            #observation, reward, done, info = 0., 0., False, {}
            

            # check if agent goes out of the frame.

        if not self.agent.alive:
            # agent died
            death = -1
            done = True
        else:
            death = 0
            done = False 
            
        goal_reached = 0
        if len(self.history_observations) > 0:
            if self.history_observations[-1][2] * 500 < 5:
                goal_reached = 2
                done = True

        reward = goal_reached + distance_reward + death + out_of_frame + collision + time_reward#+ speed_reward
        #reward = 0
        
        info = {}

        if len(self.history_observations) < self.N_PAST_TIMESTEPS:
            self.history_observations = np.zeros((self.N_internals + self.N_RAYS))  ## * self.N_PAST_TIMESTEPS)
        else:
            self.history_observations = self.history_observations[-1]

        

        truncated = False

        if self.steps > 300:
            # agent dies
            death = -1
            truncated = True
            done = True


        terminated = done
        #terminated = False
        return self.history_observations, reward, terminated, truncated, info
    
    def render(self):

        if self.rendering == "human":
            self.window.fill((0,0,0))

            # dont care about background image so far
            #self.window.blit(self.bg_image, (0,0))


            #surf = pygame.surfarray.make_surface(self.img_semantic.T)
            #self.window.blit(surf, (0, 0))

            
            surf = pygame.surfarray.make_surface(self.occupancy_grid)
            surf.set_alpha(150)
            
            #print(self.img_reference.shape)
            #surf_reference = pygame.surfarray.make_surface(np.swapaxes(self.img_reference, 0,1 ) )
            #surf_reference = pygame.surfarray.make_surface(np.swapaxes(self.img_reference, 0,1 ) )
            surf_reference = pygame.surfarray.make_surface(self.img_reference)
            self.window.blit(surf_reference, (0,0))
            self.window.blit(surf, (0, 0))

            
            # draw observation
            draw_rays = True
            if len(self.observations) > 0 and draw_rays:
                for target_pos in self.observations:
                    pygame.draw.line(self.window, (0,100,0), (self.agent.x, self.agent.y), target_pos, 1)
                    
                    
            ### Draw the agent

            # draw agent
            pygame.draw.circle(self.window, self.agent.color, (int(self.agent.x), int(self.agent.y)), 5, 3)
            #pygame.draw.rect(self.window, (0, 200, 200), pygame.Rect(int(self.x), int(self.y), 20, 10))
            
            # draw agent's goal position
            pygame.draw.circle(self.window, (255,255,0), (self.agent.goal[0], self.agent.goal[1]), 5, 3)

            # draw orientation indicator
            #p1 = (self.agent.x - 3 * np.cos(self.agent.ang), self.agent.y + 3 * np.sin(self.agent.ang))
            #p2 = (self.agent.x + 10 * np.cos(self.agent.ang), self.agent.y - 10 * np.sin(self.agent.ang))
            #pygame.draw.line(self.window,(0,100,100),p1,p2,2)

            

            pygame.display.update()
        

def pressed_to_action(keytouple):
    action_turn = 0.
    action_acc = 0.

    if keytouple[pygame.K_DOWN] == 1:  # back
        action_acc -= 1
    if keytouple[pygame.K_UP] == 1:  # forward
        action_acc += 1
    if keytouple[pygame.K_LEFT] == 1:  # left  is -1
        action_turn += 1
    if keytouple[pygame.K_RIGHT] == 1:  # right is +1
        action_turn -= 1

    if (keytouple[pygame.K_UP] == 1) and (keytouple[pygame.K_DOWN] == 1):
        action_acc = 0
    # ─── KEY IDS ─────────
    # arrow forward   : 273
    # arrow backwards : 274
    # arrow left      : 276
    # arrow right     : 275
    return np.array([action_acc, action_turn])


if __name__ == "main":
    environment = CustomEnv(dataset_name="eth", scene_name="eth", data_root="", scale_factor=1, n_agents=1, rendering="human", action_scaler=None)
    environment.init_render()

    run = True
    while run:
        # set game speed to 30 fps
        environment.clock.tick(25)   ### ETH is 2.5fps
        # ─── CONTROLS ───────────────────────────────────────────────────────────────────
        # end while-loop when window is closed
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == pygame.QUIT:
                run = False
        # get pressed keys, generate action
        get_pressed = pygame.key.get_pressed()
        action = pressed_to_action(get_pressed)


        history_observations, reward, a,b, info = environment.step(action)
        #print(len(history_observations[-1]))
        #print("internals: ", history_observations[:5])
        #print("externals: ", history_observations[5:20])
        
        print("reward:", reward)

        if a==True:
            print("done")
            break
        #print(history_observations.shape)
        # render current state
        environment.render()
        # pump event queue?
        pygame.event.pump()

    pygame.quit()



# ### Alright, now we can generate an agent with orientation that moves in an environment. This is a great starting point.
# # The actual movement dynamics of this agent are trash. Find a better way to control the agent.
# # -> The movement in counter strike is pretty nice. 
# # -> CS:Go forward,backward,left,right acceleration + angular acceleration
# # -> Stopping can be an extra action, or you press two keys in opposite direction at the same time to stop.

# ### Now we need to lift this into
# ### a) compute observations for the agent 
# ### b) add a background as a grid
# ### c) compute colisions between elements in the env
# ### d) implement policy that can control an agent -> either real policy, or pre-defined positions
# ### e) create an actual multi-agent env -> multiple agents that are governed by a policy.
