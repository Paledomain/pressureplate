import gym
from gym import spaces
import numpy as np
from enum import IntEnum
from .assets import LINEAR, CUSTOM

# Global elements
_LAYER_AGENTS = 0
_LAYER_WALLS = 1
_LAYER_DOORS = 2
_LAYER_PLATES = 3
_LAYER_GOAL = 4
_LAYER_ZONE = 5

def distance(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])




class Actions(IntEnum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Noop = 4


class Entity:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class Agent(Entity):
    def __init__(self, id, x, y, agent_plate_id = None, duration = None, zone = None):
        super().__init__(id, x, y)
        self.agent_plate_id = agent_plate_id
        self.duration = duration
        self.zone = zone

class Plate(Entity):
    def __init__(self, id, x, y, agent_plate_id = None, plate_door_id = None):
        super().__init__(id, x, y)
        self.pressed = False
        self.agent_plate_id = agent_plate_id
        self.plate_door_id = plate_door_id

# Countdown added - Changed
class Door(Entity):
    def __init__(self, id, x, y, agent_door_id = None ,plate_door_id = None, zone = None):
        super().__init__(id, x, y)
        self.open = False
        self.openCountDown = 0
        self.plate_door_id = plate_door_id 
        self.agent_door_id = agent_door_id
        self.zone = zone


class Wall(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)


class Goal(Entity):
    def __init__(self, id, x, y):
        super().__init__(id, x, y)
        self.achieved = False


class PressurePlate(gym.Env):
    """"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, height, width, n_agents, sensor_range, layout, mode):
        self.grid_size = (height, width)
        self.n_agents = n_agents
        self.sensor_range = sensor_range

        self.grid = np.zeros((6, *self.grid_size))

        self.action_space = spaces.Tuple(tuple(n_agents * [spaces.Discrete(len(Actions))]))

        self.observation_space_dim = sensor_range * sensor_range + 11

        self.observation_space = spaces.Tuple(tuple(
        n_agents * [spaces.Box(np.array([0] * self.observation_space_dim), np.array([27] * self.observation_space_dim))]
        ))

        self.cps = [(10,19), (2,14), (10, 9), (6, 2), (2,5), (3,7)]

        
        self.agents = []
        self.plates = []
        self.walls = []
        self.doors = []
        self.goal = None

        self._rendering_initialized = False
        if layout == 'linear':
            if self.n_agents == 4:
                self.layout = LINEAR['FOUR_PLAYERS']

            elif self.n_agents == 5:
                self.layout = LINEAR['FIVE_PLAYERS']

            elif self.n_agents == 6:
                self.layout = LINEAR['SIX_PLAYERS']

            elif self.n_agents == 1:
                self.layout = LINEAR['ONE_PLAYER']
            else:
                raise ValueError(f'Number of agents given ({self.n_agents}) is not supported.')
        elif layout == "custom":
            self.layout = dict(CUSTOM["FOUR_PLAYERS"])
            self.layout["AGENTS"] = self.layout["AGENTS"][mode]

        self.max_dist = np.linalg.norm(np.array([0, 0]) - np.array([2, 8]), 1)
        self.agent_order = list(range(n_agents))
        self.viewer = None

        self.room_boundaries = np.unique(np.array(self.layout['WALLS'])[:, 1]).tolist()[::-1]
        self.room_boundaries.append(-1)

    

    def step(self, actions):
        """obs, reward, done info"""
        np.random.shuffle(self.agent_order)

        for i in self.agent_order:
            proposed_pos = [self.agents[i].x, self.agents[i].y]

            if actions[i] == 0:
                proposed_pos[1] -= 1
                if not self._detect_collision(proposed_pos):
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 0 #remove agent's old position from the grid
                    self.agents[i].y -= 1
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 1 #add the agent's new position to the grid
                    self.agents[i].zone = int(self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x])
                    

            elif actions[i] == 1:
                proposed_pos[1] += 1
                if not self._detect_collision(proposed_pos):
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 0
                    self.agents[i].y += 1
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x] = 1
                    self.agents[i].zone = int(self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x])

            elif actions[i] == 2:
                proposed_pos[0] -= 1
                if not self._detect_collision(proposed_pos):
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 0
                    self.agents[i].x -= 1
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 1
                    self.agents[i].zone = int(self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x])

            elif actions[i] == 3:
                proposed_pos[0] += 1
                if not self._detect_collision(proposed_pos):
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 0
                    self.agents[i].x += 1
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 1
                    self.agents[i].zone = int(self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x])

            else:
                self.agents[i].zone = int(self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x])
                pass
        
        for i, door in enumerate(self.doors):
            if door.open is True:
                door.openCountDown = door.openCountDown - 1
                door.open = (door.openCountDown > 0)
                if door.open:
                    self.grid[_LAYER_DOORS, door.y, door.x] = 2
                else:
                    self.grid[_LAYER_DOORS, door.y, door.x] = 1

        for plate in self.plates:
            plate.pressed = False
            for agent in self.agents:
                for door in self.doors:
                    if plate.x == agent.x and plate.y == agent.y:
                        if plate.agent_plate_id == agent.agent_plate_id:
                            plate.pressed = True
                            if plate.plate_door_id == door.plate_door_id:
                                door.open = True
                                self.grid[_LAYER_DOORS, door.y, door.x] = 2
                                door.openCountDown = max(door.openCountDown, agent.duration)
                                

        # Detecting goal completion
        dones = []
        for agent in self.agents:
            agent_done = False
            agent_pos = (agent.x, agent.y)
            target_pos = self.cps[-1]
            if self.grid[_LAYER_ZONE, agent.y, agent.x] == 6 and agent.y > 6:
                agent_done = True
            dones.append(agent_done)

        return self._get_obs(), self._get_rewards_v2(dones), dones, {}

    def _detect_collision(self, proposed_position):
        """Need to check for collision with (1) grid edge, (2) walls, (3) closed doors (4) other agents"""
        # Grid edge
        if np.any([
            proposed_position[0] < 0,
            proposed_position[1] < 0,
            proposed_position[0] >= self.grid_size[1],
            proposed_position[1] >= self.grid_size[0]
        ]):
            return True

        # Walls
        for wall in self.walls:
            if proposed_position == [wall.x, wall.y]:
                return True

        # Closed Door
        for door in self.doors:
            if not door.open:
                for j in range(len(door.x)):
                    if proposed_position == [door.x[j], door.y[j]]:
                        return True

        # Other agents
        for agent in self.agents:
            if proposed_position == [agent.x, agent.y]:
                return True
            
        if proposed_position == self.goal:
            return True

        return False

    def reset(self, start_positions = None):
        # Grid wipe
        self.grid = np.zeros((6, *self.grid_size))

        #zones
        for i in range(1,7):
            for j in range(len(self.layout[f"ZONE{i}"])):
                self.grid[_LAYER_ZONE, self.layout[f"ZONE{i}"][j][1], self.layout[f"ZONE{i}"][j][0]] = i
        
        
        # Walls
        self.walls = []
        for i, wall in enumerate(self.layout['WALLS']):
            self.walls.append(Wall(i, wall[0], wall[1]))
            self.grid[_LAYER_WALLS, wall[1], wall[0]] = 1

        # Doors
        self.doors = []
        for i, door in enumerate(self.layout['DOORS']):
            self.doors.append(Door(i, door[0][0], door[0][1], agent_door_id = door[1], plate_door_id = door[2], zone = int(self.grid[_LAYER_ZONE, door[0][1][0], door[0][0][0]]) ))
            for j in range(len(door[0][0])):
                self.grid[_LAYER_DOORS, door[0][1][j], door[0][0][j]] = 1

        # Plate
        self.plates = []
        for i, plate in enumerate(self.layout['PLATES']):
            self.plates.append(Plate(i, plate[0][0], plate[0][1], agent_plate_id = plate[1], plate_door_id = plate[2]))
            self.grid[_LAYER_PLATES, plate[0][1], plate[0][0]] = 1

        # Goal
        self.goal = []
        self.goal = Goal('goal', self.layout['GOAL'][0][0], self.layout['GOAL'][0][1])
        self.grid[_LAYER_GOAL, self.layout['GOAL'][0][1], self.layout['GOAL'][0][0]] = 1

        
        
        # Agents
        self.agents = []
        for i in range(self.n_agents):
            pos, agent_plate_id, duration = self.layout['AGENTS'][i]
            if start_positions is not None:
                pos = start_positions[i]
            self.agents.append(Agent(i, pos[0], pos[1], agent_plate_id = agent_plate_id , duration = duration, zone = int(self.grid[_LAYER_ZONE, pos[1], pos[0]])))
            self.grid[_LAYER_AGENTS, pos[1], pos[0]] = 1
        return self._get_obs(), dict()

    def _get_obs(self):
        obs = []

        for i,agent in enumerate(self.agents):
            x = agent.x
            y = agent.y
            pad = self.sensor_range // 2

            x_left = max(0, x - pad)
            x_right = min(self.grid_size[1] - 1, x + pad)
            y_up = max(0, y - pad)
            y_down = min(self.grid_size[0] - 1, y + pad)

            x_left_padding = pad - (x - x_left)
            x_right_padding = pad - (x_right - x)
            y_up_padding = pad - (y - y_up)
            y_down_padding = pad - (y_down - y)

            # When the agent's vision, as defined by self.sensor_range, goes off of the grid, we
            # pad the grid-version of the observation. For all objects but walls, we pad with zeros.
            # For walls, we pad with ones, as edges of the grid act in the same way as walls.
            # For padding, we follow a simple pattern: pad left, pad right, pad up, pad down
            # Agents
            _agents = self.grid[_LAYER_AGENTS,y_up:y_down + 1, x_left:x_right + 1]

            _agents = np.concatenate((np.zeros((_agents.shape[0], x_left_padding)), _agents), axis=1)
            _agents = np.concatenate((_agents, np.zeros((_agents.shape[0], x_right_padding))), axis=1)
            _agents = np.concatenate((np.zeros((y_up_padding, _agents.shape[1])), _agents), axis=0)
            _agents = np.concatenate((_agents, np.zeros((y_down_padding, _agents.shape[1]))), axis=0)
            

            # Walls
            _walls = self.grid[_LAYER_WALLS,y_up:y_down + 1, x_left:x_right + 1]

            _walls = np.concatenate((np.ones((_walls.shape[0], x_left_padding)), _walls), axis=1)
            _walls = np.concatenate((_walls, np.ones((_walls.shape[0], x_right_padding))), axis=1)
            _walls = np.concatenate((np.ones((y_up_padding, _walls.shape[1])), _walls), axis=0)
            _walls = np.concatenate((_walls, np.ones((y_down_padding, _walls.shape[1]))), axis=0)
            

            # Doors
            _doors = self.grid[_LAYER_DOORS,y_up:y_down + 1, x_left:x_right + 1]

            _doors = np.concatenate((np.zeros((_doors.shape[0], x_left_padding)), _doors), axis=1)
            _doors = np.concatenate((_doors, np.zeros((_doors.shape[0], x_right_padding))), axis=1)
            _doors = np.concatenate((np.zeros((y_up_padding, _doors.shape[1])), _doors), axis=0)
            _doors = np.concatenate((_doors, np.zeros((y_down_padding, _doors.shape[1]))), axis=0)
            

            # Plate
            _plates = self.grid[_LAYER_PLATES,y_up:y_down + 1, x_left:x_right + 1]

            _plates = np.concatenate((np.zeros((_plates.shape[0], x_left_padding)), _plates), axis=1)
            _plates = np.concatenate((_plates, np.zeros((_plates.shape[0], x_right_padding))), axis=1)
            _plates = np.concatenate((np.zeros((y_up_padding, _plates.shape[1])), _plates), axis=0)
            _plates = np.concatenate((_plates, np.zeros((y_down_padding, _plates.shape[1]))), axis=0)
           

            # Goal
            _goal = self.grid[_LAYER_GOAL, y_up:y_down + 1, x_left:x_right + 1]

            _goal = np.concatenate((np.zeros((_goal.shape[0], x_left_padding)), _goal), axis=1)
            _goal = np.concatenate((_goal, np.zeros((_goal.shape[0], x_right_padding))), axis=1)
            _goal = np.concatenate((np.zeros((y_up_padding, _goal.shape[1])), _goal), axis=0)
            _goal = np.concatenate((_goal, np.zeros((y_down_padding, _goal.shape[1]))), axis=0)
            
            agent_obs = _agents + 2 * _walls + 3 * _goal + 4 * _plates + 6 * _doors
            agent_obs = agent_obs.reshape(-1) #in production uncomment this line and the next one, this version better for human readability
            
            agent_pos = (agent.x, agent.y)

            plate_dists = []
            for plate in self.plates:
                if plate.agent_plate_id == agent.agent_plate_id:
                    plate_dists.append(distance((plate.x, plate.y), agent_pos))
            closest_plate = min(plate_dists)

            door_dists = []
            for door in self.doors:
                door_dists.append(distance((door.x[0], door.y[0]), agent_pos))
            door_dists = np.array(door_dists)
            cp_distance = distance(self.cps[agent.zone - 1], agent_pos)

            if i == 0:
                distance_to_other = distance((self.agents[1].x, self.agents[1].y), agent_pos)
            elif i == 1: 
                distance_to_other = distance((self.agents[0].x, self.agents[0].y), agent_pos)
            agent_obs = np.concatenate((agent_obs, door_dists, np.array([agent.y, agent.x, agent.zone, cp_distance, distance_to_other, closest_plate])))
            obs.append(agent_obs)
            #state representation: empty:0, agent:1, wall:2, goal:3, plate_and_no_agent:4, plate_and_agent:5, 
            #closed_door_and_no_agent:6, closed_door_and_agent:7, open_door_and_no_agent:12, open_door_and_agent:13
            #door layout changes based on the status of the doors. if door is open it is labeled with 2 otherwise 1
        return tuple(obs)

    def _get_flat_grid(self):
        grid = np.zeros(self.grid_size)

        # Plate
        for plate in self.plates:
            grid[plate.y, plate.x] = 2

        # Walls
        for wall in self.walls:
            grid[wall.y, wall.x] = 3

        # Doors
        for door in self.doors:
            if door.open:
                grid[door.y, door.x] = 0
            else:
                grid[door.y, door.x] = 4

        # Goal
        grid[self.goal.y, self.goal.x] = 5

        # Agents
        for agent in self.agents:
            grid[agent.y, agent.x] = 1

        return grid
    
    def _get_rewards_v2(self, dones):
        rewards = []        

        # The last agent's desired location is the goal instead of a plate, so we use an if/else block
        # to break between the two cases
        any_ = False
        is_door_open = self.grid[_LAYER_DOORS][22,10] == 2

        for i, agent in enumerate(self.agents):
            if i == 0:
                least_zone = agent.zone
            else:
                least_zone = min(agent.zone, least_zone)

        for i, agent in enumerate(self.agents):
            agent_pos = (agent.x, agent.y)
            door_reward = 0
            agent_reward = 0
            zone_reward = -(5 - agent.zone) - (5 - least_zone) #negative zone reward
            for door in self.doors:
                if door.openCountDown > 0:
                    agent_id = door.agent_door_id - 1
                    if agent_id == i:
                        if self.agents[0].zone == door.zone and self.agents[1].zone == door.zone:
                            door_reward = 1

            distance_reward = - distance(self.cps[agent.zone - 1], agent_pos)

            agent_reward = zone_reward * (5/1) + door_reward * (2.2) + distance_reward * (1/4) + 35
            if dones[i]:
                agent_reward += 10000           
            rewards.append(agent_reward)
        return rewards
    """
if i == 0:
    if self.grid[_LAYER_DOORS][22,10] == 2:
        if agent.zone == 1 and self.agents[1].zone == 1:
            door_reward = 1
    if self.grid[_LAYER_DOORS][15,2] == 2:
        if agent.zone == 2 and self.agents[1].zone == 2:
            door_reward = 1"""

    def _get_rewards_dummy(self):
        agent0_target = (13,23)
        agent1_target = (4,23)
        agent2_target = (4,27)
        agent3_target = (13,27)
        rewards = []

        def f(a,b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        # The last agent's desired location is the goal instead of a plate, so we use an if/else block
        # to break between the two cases
        for i, agent in enumerate(self.agents):
            agent_pos = (agent.x, agent.y)
            if i == 0:
                reward = -f(agent_pos, agent0_target) / 10
                if reward == 0:
                    reward = 10
                rewards.append(reward)
            elif i ==1:
                reward = -f(agent_pos, agent1_target) / 10
                if reward == 0:
                    reward = 10
                rewards.append(reward)
            elif i == 2:
                reward = -f(agent_pos, agent2_target) / 10
                if reward == 0:
                    reward = 10
                rewards.append(reward)
            elif i == 3:
                reward = -f(agent_pos, agent3_target) / 10
                if reward == 0:
                    reward = 10
                rewards.append(reward)
        return rewards
    
    

    def _init_render(self):
        from .rendering import Viewer
        self.viewer = Viewer(self.grid_size)
        self._rendering_initialized = True

    def render(self, mode='human'):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
