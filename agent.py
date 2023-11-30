import numpy as np

import random

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def cross_front(player_front_position, agent_position):                 # These two functions are defined to know
    return np.array_equal(agent_position, player_front_position)        # wether the agent touched the front or the back 
                                                                        # of the player.
def cross_back(player_back_position, agent_position):                   # If it touched the front, the bypass is unsuccessful
    return np.array_equal(agent_position, player_back_position)         # if it touched the back, the player is bypassed

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}      #The environment supports a rendering mode specifically designed for human visualization (continuous display).

    def __init__(self):
        super(CustomEnv, self).__init__()

        self.action_space = spaces.Discrete(4) #We only have four actions which a discrete value : {up, down, left, right}
        self.observation_space = spaces.Box(low=-12, high=12, shape=(5, 2), dtype=np.int32) #Defines the observation space for our custom env, the observations consist of a 5x2 grid of integers ranging from -12 to 12.

    def step(self, action):

        #Calculating the two distances : player's front - agent | player's back - agent
        #This distance will be used in our rewarding system.
        dist_agent_player_front = np.linalg.norm(self.agent_position - self.player_front_position)
        dist_agent_player_back = np.linalg.norm(self.agent_position - self.player_back_position)
        #Randomly initializing the agent's position 
        new_position = self.agent_position

        if action == 1:
            # Go right
            new_position = np.array([self.agent_position[0], self.agent_position[1] + 1])
        elif action == 2:
            # Go left
            new_position = np.array([self.agent_position[0], self.agent_position[1] - 1])
        elif action == 3:
            # Go down
            new_position = np.array([self.agent_position[0] + 1, self.agent_position[1]])
        elif action == 4:
            # Go up
            new_position = np.array([self.agent_position[0] - 1, self.agent_position[1]])

            # First check if the position is valid : Within boundaries and not an obstacle
        if 0 <= new_position[0] < 12 and 0 <= new_position[1] < 12 and self.grid[new_position[0], new_position[1]] != 1:
            self.agent_position = new_position
            # Penalize the agent for hitting an obstacle
            self.reward = -100
        else:
            # If the position invalid, try random directions until a valid one is found
            valid_direction_found = False
            while not valid_direction_found:
                random_action = random.choice([1, 2, 3, 4])
                new_position = self.agent_position

                if random_action == 1:
                    # Move right
                    new_position[1] += 1
                elif random_action == 2:
                    # Move left
                    new_position[1] -= 1
                elif random_action == 3:
                    # Move down
                    new_position[0] += 1
                elif random_action == 4:
                    # Move up
                    new_position[0] -= 1

                # Check if the new position is within boundaries and not an obstacle
                if 0 <= new_position[0] < 12 and 0 <= new_position[1] < 12 and self.grid[new_position[0], new_position[1]] != 1:
                    self.agent_position = new_position
                    valid_direction_found = True

        # If the agent steps on the back or front of the player, end the episode with messages
        if cross_front(self.player_front_position, self.agent_position):
            #Penalize the agent
            self.reward = -100
            self.done = True
            print("Bypass unsuccessful")
        elif cross_back(self.player_back_position, self.agent_position):
            #reward the agent
            self.reward = 100
            self.done = True
            print("Bypassed")
        else:
            #keep rrewarding according to the distances : if the agent is closer to the front than to the back,
            #the agent is penalized, but if it's closer to the back than to the front, the agent is rewarded
            self.reward = dist_agent_player_front - dist_agent_player_back

        # Displaying the grid, the agent, the player  
        plt.imshow(self.grid, cmap='gray')
        plt.scatter(self.player_position[1], self.player_position[0], c='b', marker='o', s=100, edgecolor='k')
        plt.scatter(self.player_front_position[1], self.player_front_position[0], c='r', marker='o', s=50, edgecolor='k')
        plt.scatter(self.player_back_position[1], self.player_back_position[0], c='g', marker='o', s=50, edgecolor='k')
        plt.scatter(self.agent_position[1], self.agent_position[0], c='pink', marker='o', s=100, edgecolor='k')
        plt.show(block=False)
        #time to see the progress
        plt.pause(0.1)
        plt.clf()

        dist_agent_player_front = np.absolute(self.agent_position - self.player_front_position)
        dist_agent_player_back = np.absolute(self.agent_position - self.player_back_position)
        
        #the observatiçon vector is used to update the state perceived by the RL agent during its interaction with the env
        observation = np.array([self.agent_position, self.player_back_position, self.player_front_position,
                                dist_agent_player_front, dist_agent_player_back])
        
        info = {} #Since we did not used the info and truncated we set them to insignificant values
        return observation, self.reward, self.done,False, info 

    def reset(self, num_obstacles=None, seed=None):
        #We start by setting every attribute we need 
        self.done = False
        self.reward = -10
        num_obstacles = num_obstacles if num_obstacles is not None else np.random.randint(50, 80)
        self.grid = self.random_obstacles(num_obstacles)
        self.agent_position = np.array(self.random_position())
        self.player_position = np.array(self.random_position(exclude_position=self.agent_position))
        self.player_back_position = np.array(self.random_player_back_position())
        self.player_front_position = np.array(self.random_player_front_position(self.player_back_position))

        dist_agent_player_front = np.absolute(np.array(self.agent_position) - np.array(self.player_front_position))
        dist_agent_player_back = np.absolute(np.array(self.agent_position) - np.array(self.player_back_position))
        #the observatiçon vector is used to update the state perceived by the RL agent during its interaction with the env
        observation = np.array([self.agent_position, self.player_back_position, self.player_front_position,
                                dist_agent_player_front, dist_agent_player_back])
        info = {}
        return observation, info

    #Function for display

    def render(self, mode='human'):
        plt.imshow(self.grid, cmap='gray')
        plt.show()
    #Function to randomly set obstacles on the grid, the 1 are obstacles (the obstacles are white)
    def random_obstacles(self, num_obstacles):
        grid_with_obstacles = np.zeros((12, 12), dtype='uint8')

        for _ in range(num_obstacles):
            obstacle_position = [np.random.randint(12), np.random.randint(12)]
            grid_with_obstacles[obstacle_position[0], obstacle_position[1]] = 1

        return grid_with_obstacles
    #Function to randomly assign a position to the agent/player
    def random_position(self, exclude_position=None):
        while True:
            random_position = [np.random.randint(12), np.random.randint(12)]
            if (
                self.grid[random_position[0], random_position[1]] != 1
                and (exclude_position is None or not np.any(random_position == exclude_position))
            ):
                return random_position
    #Function to randomly chose the back of the player ( the winning point )
    def random_player_back_position(self):
        while True:
            direction = random.choice(["top", "bottom", "left", "right"])
            if direction == "top" and self.player_position[0] - 1 >= 0:
                player_back_position = [self.player_position[0] - 1, self.player_position[1]]
            elif direction == "bottom" and self.player_position[0] + 1 < 12:
                player_back_position = [self.player_position[0] + 1, self.player_position[1]]
            elif direction == "left" and self.player_position[1] - 1 >= 0:
                player_back_position = [self.player_position[0], self.player_position[1] - 1]
            elif direction == "right" and self.player_position[1] + 1 < 12:
                player_back_position = [self.player_position[0], self.player_position[1] + 1]
            else:
                continue

            if (
                self.grid[player_back_position[0], player_back_position[1]] != 1
                and 0 <= player_back_position[0] < 12
                and 0 <= player_back_position[1] < 12
            ):
                return player_back_position
    # determine the front position based on the direction of the back position (not random)
    def random_player_front_position(self, back_position):
        
        back_x, back_y = back_position
        front_x, front_y = back_x, back_y  # Default to the same position

        if back_y > self.player_position[1]:
            front_x, front_y = self.player_position[0], self.player_position[1] - 1
        elif back_y < self.player_position[1]:
            front_x, front_y = self.player_position[0], self.player_position[1] + 1
        elif back_x > self.player_position[0]:
            front_x, front_y = self.player_position[0] - 1, self.player_position[1]
        elif back_x < self.player_position[0]:
            front_x, front_y = self.player_position[0] + 1, self.player_position[1]

        # check if the front position is valid
        if (
            0 <= front_x < 12
            and 0 <= front_y < 12
            and self.grid[front_x, front_y] != 1
        ):
            return [front_x, front_y]
        else:
            return back_position

#We create a vectorized env with our custom env
env = DummyVecEnv([lambda: CustomEnv()])
# Create a DQN model with the "MlpPolicy" and the specified env, and enable verbose loggin. We then train it for 10k steps witk logging every 1 step
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=1)
model.save("dqn_cartpole") #We save it in a file

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole") #reload it from the file

obs, info = env.reset() # Reset the env and obtain the initial observation and information
while True:
    action, _states = model.predict(obs, deterministic=True)
    # Predict the action using the trained model and the current observation. If the episode is done we reset the env and go for another ep
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()