# -*- coding: utf-8 -*-
#from gym import Env, spaces
import gymnasium as gym
import pygame
import numpy as np
import distinctipy
import json
import cv2
import time
import distinctipy

from src.agents.agent import PedestrianAgent

rotation_max, acceleration_max = 0.08, 0.5


def circle_mask(x_cells, y_cells, cx, cy, r):
    mask = (x_cells[np.newaxis,:]- cx)**2 + (y_cells[:,np.newaxis]-cy)**2 < r**2
    return mask


from src.data.scene_loader import TrajScene

class MultiAgentEnv(gym.Env):
    def __init__(self, dataset_name, scene_name, data_root, scale_factor=1, rendering="human", 
    action_scaler=None, n_rays=32, env_config={}):
        self.agent_colors = [tuple(np.array(col) * 255) for col in distinctipy.get_colors(100)]
        
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
        self.done_agents = []

        # "Real" Env Agents
        self.MAX_VELOCITY = 5
        self.N_PAST_TIMESTEPS = 2
        self.N_RAYS = n_rays
        self.N_internals = 4

        # init crowd randomly at creation of the env.
        self.init_crowd()

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

    
    def init_crowd(self, random=True, aid=None):

        # We could also sample random agents => random choice without replacement on the agent ids over the whole agent set.
        # mini reset env
        self.agents = []
        self.occupancy_grid = []
        self.observations = []
        self.init_environment_map()

        # Initialize a crowd, i.e., a set of agents.
        if not random:

            # We initialize based on some agent. 
            # Get the agent's start and goal position.
            #s = self.trajs[aid][["pos_x", "pos_y"]].values
            #pos, goal = self.sample_agent(replay_agent=True, replay_pos=s[0], replay_goal=s[-1])

            # Get the init frame and corresponding dataframe.
            self.replay_start_frame = self.trajs_df[self.trajs_df.agent_id == aid].frame_id.values[0]
            #self.replay_start_frame = self.replay_start_frame + self.steps
        else:
            _, _, aid = self.sample_agent()
            self.replay_start_frame = self.trajs_df[self.trajs_df.agent_id == aid].frame_id.values[0]


        # Get all agents that are present in the current frame.
        temp = self.trajs_df[self.trajs_df.frame_id == self.replay_start_frame]
        crowd_agent_ids = temp.agent_id.unique()

        #print(crowd_agent_ids)
        for cid in crowd_agent_ids:
            # get the dataframe
            temp = self.trajs[cid]
            # get all positions starting at the current frame.
            s = temp[temp.frame_id >= self.replay_start_frame][["pos_x", "pos_y"]].values
            
            # get "start" and "goal" position of this agent
            pos, goal = self.sample_agent(replay_agent=True, replay_pos=s[0], replay_goal=s[-1])

            #print(pos, goal)
            ag = PedestrianAgent(x=pos[0], y=pos[1], goal=goal, color=self.agent_colors[len(self.agents)], 
                                        radius=3, max_vel=self.MAX_VELOCITY, lidar_num_rays=self.N_RAYS, 
                                        max_x = self.window_width, max_y=self.window_height, aid=aid, action_scaler=self.action_scaler)

            # init agent on the occupancy grid
            mask = circle_mask(self.x_cells, self.y_cells, ag.y, ag.x, ag.circle.radius)
            self.occupancy_grid[mask] = ag.id

            # Add to list of agents.
            self.agents.append(ag)

        print("n_agents = ", len(self.agents))

    # Do not allow to have a crowd in the background.
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


    def step(self, actions):
        ### Perform one step with actions
        #for agent, action in zip(self.agents, actions):
        # should return new observation etc., then we make a list

        # Step for each agent.
        for agent, action in zip(self.agents, actions):
            agent.dynamics_step(action)

        self.steps += 1

        new_agent_masks = []
        for agent in self.agents:
            # Check collision.
            agent_mask = circle_mask(self.x_cells, self.y_cells, agent.y, agent.x, agent.circle.radius)
            new_agent_masks.append(agent_mask)
            collision = 0
            mask_temp = self.occupancy_grid == agent.id
            self.occupancy_grid[mask_temp] = 0 # remove agent from map, check if new collision collides with anything in env

            if np.sum(self.occupancy_grid[np.where(agent_mask == True )]) > 0:
                agent.kill()
                self.done_agents.append(agent)                

            # Check for out of frame
            #out_of_frame = 0
            if (agent.y >= self.window_width) or (agent.x >= self.window_height) or (agent.x < 0) or (agent.y < 0):
                agent.kill()
                self.done_agents.append(agent)

            # draw agent on the map again because need to check for other agents collisions.
            self.occupancy_grid[mask_temp] = agent.id 
        
        # Check if agent reached the goal:
        for agent in self.agents:

            if agent.dist_goal <= 0.01: #0.005:
                agent.done()
                self.done_agents.append(agent)

        self.agents = [agent for agent in self.agents if agent.alive and (not agent.finished)]
        
        # Update occupancy grid with new agent's position
        # Clear all agents
        self.occupancy_grid[self.occupancy_grid >= 10] = 0

        for agent, agent_mask in zip(self.agents, new_agent_masks):
            # Set agent
            self.occupancy_grid[agent_mask] = agent.id
        

        # Get observations
        self.observations = []
        self.ray_observations = []
        for agent in self.agents:
            obs, ray_points, _ = agent.get_observation(self.occupancy_grid)
            self.observations.append(obs[-1])
            self.ray_observations.append(ray_points)

        reward = 0

        done = False

        if len(self.agents) == 0:
            done = True

        truncated = False
        if self.steps > 300:
            truncated = True
            done = True

        terminated = done
        #terminated = False

        n_survived = 0
        for agent in self.done_agents:
            if agent.alive:
                n_survived += 1
        if len(self.done_agents) > 0:
            info = {"num_survivors" : n_survived, "survival_rate" : n_survived / len(self.done_agents) }
        else:
            info = {}
        return self.observations, reward, terminated, truncated, info
    

    def init_render(self):
        # use pygame for rendering
        pygame.init()
        self.window = pygame.display.set_mode((self.window_height, self.window_width) )
        self.clock = pygame.time.Clock()
        # call render once to render all the initial things so 
        # that we can use  them directly to compute the first observation
        self.render()
    

    def reset(self, seed=0, random=True, aid=None):
        # reset the environment to initial state
        self.agents = []
        self.done_agents = []
        self.occupancy_grid = []
        self.observations = []
        self.ray_observations = []

        self.init_environment_map()
        #self.replay_agent_loader = ETH_Agent_Loader()

        if not random:
            self.init_crowd(random=random, aid=aid)
        else:
            self.init_crowd()

        self.steps = 0

        # reset pygame if rendering = human
        if self.rendering == "human":
            pygame.quit()
            pygame.init()
            self.window = pygame.display.set_mode((self.window_height, self.window_width))
            self.clock = pygame.time.Clock()
            self.render()
        
        
        #return observation
        observations = []
        for agent in self.agents:
            obs_0, _, _ = agent.get_observation(self.occupancy_grid)
            observations.append(np.array(obs_0))

        return np.vstack(observations).astype(np.float64), {}
        #return (np.array(obs_0[-1]).astype(np.float64), {})

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
            draw_rays = False
            if len(self.observations) > 0 and draw_rays:
                for agent, rays in zip(self.agents, self.ray_observations):
                    for target_pos in rays:
                        pygame.draw.line(self.window, (100,0,0), (agent.x, agent.y), target_pos, 1)
            
                    
            ### Draw the agent

            # draw agents
            for agent in self.agents + self.done_agents:
                pygame.draw.circle(self.window, agent.color, (int(agent.x), int(agent.y)), 5, 3)
            #pygame.draw.rect(self.window, (0, 200, 200), pygame.Rect(int(self.x), int(self.y), 20, 10))
            
            # draw agent's goal position
            for agent in self.agents:
                pygame.draw.circle(self.window, (255,255,0), (agent.goal[0], agent.goal[1]), 5, 3)

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
