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
    def __init__(self, id, x, y, agent_plate_id = None, duration = None):
        super().__init__(id, x, y)
        self.agent_plate_id = agent_plate_id
        self.duration = duration
        self.zone = 1

class Plate(Entity):
    def __init__(self, id, x, y, agent_plate_id = None, plate_door_id = None):
        super().__init__(id, x, y)
        self.pressed = False
        self.agent_plate_id = agent_plate_id
        self.plate_door_id = plate_door_id

# Countdown added - Changed
class Door(Entity):
    def __init__(self, id, x, y, agent_door_id = None ,plate_door_id = None):
        super().__init__(id, x, y)
        self.open = False
        self.openCountDown = 0
        self.plate_door_id = plate_door_id 
        self.agent_door_id = agent_door_id


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

    def __init__(self, height, width, n_agents, sensor_range, layout):
        self.grid_size = (height, width)
        self.n_agents = n_agents
        self.sensor_range = sensor_range

        self.grid = np.zeros((6, *self.grid_size))

        self.action_space = spaces.Tuple(tuple(n_agents * [spaces.Discrete(len(Actions))]))

        self.observation_space_dim = self.grid_size[0] * self.grid_size[1] + 2

        self.observation_space = spaces.Tuple(tuple(
        n_agents * [spaces.Box(np.array([0] * self.observation_space_dim), np.array([1] * self.observation_space_dim))]
        ))


        
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
            if self.n_agents == 4:
                self.layout = CUSTOM["FOUR_PLAYERS"]

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
                    self.agents[i].zone = self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x]
                    

            elif actions[i] == 1:
                proposed_pos[1] += 1
                if not self._detect_collision(proposed_pos):
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 0
                    self.agents[i].y += 1
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 1
                    self.agents[i].zone = self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x]

            elif actions[i] == 2:
                proposed_pos[0] -= 1
                if not self._detect_collision(proposed_pos):
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 0
                    self.agents[i].x -= 1
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 1
                    self.agents[i].zone = self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x]

            elif actions[i] == 3:
                proposed_pos[0] += 1
                if not self._detect_collision(proposed_pos):
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 0
                    self.agents[i].x += 1
                    self.grid[_LAYER_AGENTS, self.agents[i].y, self.agents[i].x ] = 1
                    self.agents[i].zone = self.grid[_LAYER_ZONE, self.agents[i].y, self.agents[i].x]

            else:
                # NOOP
                pass
        

        #CountDown added  - Changed
        #Different agents can use pressure plate  - Changed
        #for i, plate in enumerate(self.plates):
        #    #for agentId in range(0, len(self.agents)):
        #    plate_coordinates = np.array([plate.x, plate.y])
        #    agents_coordinates = np.array([[self.agents[agentId].x, self.agents[agentId].y] for agentId in range(0, len(self.agents)) ])
        #    if not plate.pressed:
        #        if any(np.all(plate_coordinates == agent_coordinates) for agent_coordinates in agents_coordinates):
        #            plate.pressed = True
        #            self.doors[plate.id].open = True
        #            if [plate.x, plate.y] == [self.agents[plate.id].x, self.agents[plate.id].y]:
        #                self.doors[plate.id].openCountDown = 6
        #            else:
        #                # Agent[plate.id] may have used before so we need to take max of it
        #                self.doors[plate.id].openCountDown = max(self.doors[plate.id].openCountDown, 3) #2
        #            break
        #    else:
        #        if not any(np.all(plate_coordinates == agent_coordinates) for agent_coordinates in agents_coordinates):
        #            plate.pressed = False
        #            self.doors[plate.id].open = (self.doors[plate.id].openCountDown > 0)
        #        else:
        #            plate.pressed = True
        #            self.doors[plate.id].open = True
        #            if [plate.x, plate.y] == [self.agents[plate.id].x, self.agents[plate.id].y]:
        #                self.doors[plate.id].openCountDown = 6
        #            else:
        #                # Agent[plate.id] may have used before so we need to take max of it
        #                self.doors[plate.id].openCountDown = max(self.doors[plate.id].openCountDown, 3) #2

         # Door Countdown  - Changed #is it okay to change door status before moving agents??????
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
        completed = True
        for agent in self.agents:
            if self.grid[_LAYER_ZONE, agent.y, agent.x] != 6:
                completed = False

        if completed:
            self.goal.achieved = True

        print( self._get_rewards() )
        return self._get_obs(), self._get_rewards(), [self.goal.achieved] * self.n_agents, {}

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

    def reset(self):
        # Grid wipe
        self.grid = np.zeros((6, *self.grid_size))

        # Agents
        self.agents = []
        for i in range(self.n_agents):
            pos, agent_plate_id, duration = self.layout['AGENTS'][self.agent_order[i]]
            
            self.agents.append(Agent(i, pos[0], pos[1], agent_plate_id = agent_plate_id , duration = duration))
            self.grid[_LAYER_AGENTS, pos[1], pos[0]] = 1

        # Walls
        self.walls = []
        for i, wall in enumerate(self.layout['WALLS']):
            self.walls.append(Wall(i, wall[0], wall[1]))
            self.grid[_LAYER_WALLS, wall[1], wall[0]] = 1

        # Doors
        self.doors = []
        for i, door in enumerate(self.layout['DOORS']):
            self.doors.append(Door(i, door[0][0], door[0][1], agent_door_id = door[1], plate_door_id = door[2]))
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

        #zones
        for i in range(1,7):
            for j in range(len(self.layout[f"ZONE{i}"])):
                self.grid[_LAYER_ZONE, self.layout[f"ZONE{i}"][j][1], self.layout[f"ZONE{i}"][j][0]] = i

        return self._get_obs()

    def _get_obs(self):
        obs = []
        """
        _agents = self.grid[_LAYER_AGENTS,:,:] #2D arrays of grid size. Entry is 1 if respective entity exists on the entry coordiante and 0 otherwise
        _walls = self.grid[_LAYER_WALLS,:,:]
        _doors = self.grid[_LAYER_DOORS,:,:]
        _plates = self.grid[_LAYER_PLATES,:,:]
        _goal = self.grid[_LAYER_GOAL,:,:]
        common_obs = _agents + 2 * _walls + 3 * _goal + 4 * _doors + 7 * _plates
        for door in self.doors:
            if door.open:
                common_obs[door.y, door.x] += 2   
        common_obs = common_obs.reshape(-1) """

        for agent in self.agents:
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
            # agent_obs = agent_obs.reshape(-1) in production uncomment this line and the next one, this version better for human readability
            # agent_obs = np.concatenate((agent_obs, np.array[agent.y, agent.x, agent.duration]))
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

    def _get_rewards(self):
        rewards = []

        # The last agent's desired location is the goal instead of a plate, so we use an if/else block
        # to break between the two cases
        for i, agent in enumerate(self.agents):

            if i == len(self.agents) - 1:
                plate_loc = self.goal.x, self.goal.y
            else:
                plate_loc = self.plates[i].x, self.plates[i].y

            curr_room = self._get_curr_room_reward(agent.y)

            agent_loc = agent.x, agent.y

            if i == curr_room:
                reward = - np.linalg.norm((np.array(plate_loc) - np.array(agent_loc)), 1) / self.max_dist
            else:
                #print(str(agent.x) + "," + str(agent.y))
                reward = -len(self.room_boundaries)+1 + curr_room
            
            rewards.append(reward)
        return rewards

    def _get_curr_room_reward(self, agent_y):
        for i, room_level in enumerate(self.room_boundaries):
            if agent_y > room_level:
                curr_room = i
                break

        return curr_room

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
