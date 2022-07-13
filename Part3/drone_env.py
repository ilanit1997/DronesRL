import random
from copy import deepcopy

DRONE_CAPACITY = 2
RESET_REWARD = -10
WAIT_REWARD = -1
MOVE_REWARD = -1
SUCC_PICK_REWARD = 1
FAILED_PICK_REWARD = -1
BAD_DELIVER_REWARD = -1
SUCCESS_REWARD = 100
WIND_PROB = (.1, .8, .1)


class Map:
    def __init__(self, map: tuple):
        self.map = map
        self._locations = None
        self._passable_locations = None

    @property
    def locations(self):
        if self._locations is None:
            self._locations = set((x, y) for x in range(len(self.map)) for y in range(len(self.map[0])))
        return self._locations

    @property
    def passable_locations(self):
        if self._passable_locations is None:
            self._passable_locations = set((x, y) for x, y in self.locations if self.map[x][y].startswith('P'))
        return self._passable_locations

    def get_wind_direction(self, x, y):
        wind_property = self.map[x][y].split('_')[1]
        assert wind_property in ['NW', 'WU', 'WD', 'WL', 'WR'], f'{self.map[x][y]} should be in [NW, WU, WD, WL, WR]'
        if wind_property == 'NW':
            return 0, 0
        elif wind_property == 'WU':
            return -1, 0
        elif wind_property == 'WD':
            return 1, 0
        elif wind_property == 'WR':
            return 0, 1
        elif wind_property == 'WL':
            return 0, -1


class Drone:
    def __init__(self, location):
        self.location = location
        self.packages = []


class Package:
    def __init__(self, name, location):
        self.name = name
        self.location = location

    def as_tuple(self):
        return self.name, self.location

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return (self.__class__ == other.__class__) & (self.as_tuple() == other.as_tuple())


class DroneEnv:
    action_space = ['reset', 'wait', 'pick', 'move_up', 'move_down', 'move_left', 'move_right', 'deliver']

    def __init__(self, params):
        self._input_params = deepcopy(params)
        self.map = Map(params['map'])
        self.packages = list(Package(name, location) for name, location in params['packages'])
        self.target_location = params['target_location']
        self.success_rate = params['success_rate']
        self.current_steps = 0
        if params['drone_location'] == 'random':
            drone_location = random.choice(tuple(self.map.passable_locations))
        else:
            drone_location = params['drone_location']
        assert drone_location in self.map.passable_locations, f'drone should be in passable locations only'
        self.drone = Drone(drone_location)

    def reset(self):
        """reset the state of the environment to an initial state"""
        self.__init__(self._input_params)
        return self._next_observation()

    def _next_observation(self):
        """get drone location, package location, goal location"""
        obs = dict(
            drone_location=self.drone.location,
            packages=set(p.as_tuple() for p in self.packages),
            target_location=self.target_location
        )
        return obs

    def step(self, action):
        """ Execute one time step within the environment """
        # sanity check
        assert action in self.action_space, f'action: {action} is not in action space: {self.action_space}'
        # execute reset action
        if action == 'reset':
            reward = self._execute_action(action)
            obs1 = self._next_observation()
            done = True
            return obs1, reward, done
        # execute action
        if random.random() < self.success_rate:
            # action succeeds
            reward = self._execute_action(action)
        else:
            # action fails
            alternative_action = random.choice(['move_up', 'move_down', 'move_left', 'move_right', 'wait'])
            reward = self._execute_action(alternative_action)
        # stochastic step
        self._stochastic_step()
        # update steps counter
        self.current_steps += 1
        # compute next observation
        obs1 = self._next_observation()
        done = action == 'reset'
        return obs1, reward, done

    def _stochastic_step(self):
        x, y = self.drone.location
        wind_direction_x, wind_direction_y = self.map.get_wind_direction(x, y)
        if sum((wind_direction_x, wind_direction_y)) == 0:
            return
        else:
            wind_power = random.choices([0, 1, 2], WIND_PROB, k=1)[0]
            if wind_power == 0:
                return
            elif wind_power == 2:
                dx = wind_direction_x * wind_power
                dy = wind_direction_y * wind_power
                if (x + dx, y + dy) in self.map.passable_locations:
                    self.drone.location = x + dx, y + dy
                    return
                else:
                    wind_power = 1  # try again with wind_power = 1
            if wind_power == 1:
                dx = wind_direction_x * wind_power
                dy = wind_direction_y * wind_power
                if (x + dx, y + dy) in self.map.passable_locations:
                    self.drone.location = x + dx, y + dy
                    return

    def _execute_action(self, action):
        if action == 'reset':
            self.reset()
            return RESET_REWARD
        elif action == 'wait':
            return WAIT_REWARD
        elif action.startswith('move'):
            x, y = self.drone.location
            if action == 'move_up':
                x -= 1
            elif action == 'move_down':
                x += 1
            elif action == 'move_left':
                y -= 1
            elif action == 'move_right':
                y += 1
            if (x, y) in self.map.passable_locations:
                self.drone.location = (x, y)
            return MOVE_REWARD
        elif action == 'pick':
            packages_at_location = [p for p in self.packages if p.location == self.drone.location]
            if packages_at_location and len(self.drone.packages) < DRONE_CAPACITY:
                package_to_pick = packages_at_location[0]
                self.drone.packages.append(package_to_pick)
                package_to_pick.location = 'drone'
                assert package_to_pick in self.packages, "env.packages should include the picked package"
                return SUCC_PICK_REWARD
            return FAILED_PICK_REWARD
        elif action == 'deliver':
            if self.drone.packages:
                package_to_deliver = self.drone.packages.pop(0)
                package_to_deliver.location = self.drone.location
                if package_to_deliver.location == self.target_location:
                    # self.packages.append(package_to_deliver)
                    self.packages.remove(package_to_deliver)
                    return SUCCESS_REWARD
                return BAD_DELIVER_REWARD
            else:
                return BAD_DELIVER_REWARD
