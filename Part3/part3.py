import random


class DroneAgent:
    def __init__(self, n, m):
        """
            obs = dict(
                drone_location=self.drone.location,
                packages=set(p.as_tuple() for p in self.packages),
                target_location=self.target_location
            )
           """
        self.mode = 'train'  # do not change this!
        self.Q = {}
        self.alpha = 0.5
        self.gamma = 0.95
        self.steps = 0
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'wait']
        self.num_pack = 0
        self.first_deliver_flag = False
        self.delivery_time = 0
        self.deliveries = []
        self.map_shape = [n, m]
        self.last_delivery = 0


    def select_action(self, obs0):
        #check what should do
        if self.can_reset(obs0):
            return "reset"

        if self.can_deliver(obs0):
            return "deliver"

        if self.can_pickup(obs0):
            return "pick"

        obs0['packages'] = tuple(obs0['packages'])
        obs0 = tuple(obs0.items())
        if self.mode == 'train' or len(self.Q.keys()) == 0 or obs0 not in self.Q.keys():
            acts = self.can_move(obs0[0][1])
            acts.append('wait')
            act = random.choice(acts)
            return act
        else:
            max_act = max(self.Q[obs0].items(), key=lambda x:x[1])[0]
            return max_act



    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        self.steps += 1
        obs0['packages'] = tuple(obs0['packages'])
        obs1['packages'] = tuple(obs1['packages'])
        if action == 'deliver' and len(obs0['packages']) != len(obs1['packages']):
            delivery_time = self.steps - self.last_delivery
            self.last_delivery = self.steps
            self.delivery_time = delivery_time if delivery_time < self.delivery_time else self.delivery_time


        obs0, obs1 = tuple(obs0.items()), tuple(obs1.items())
        q_sa, max_val = 0, 0
        if len(self.Q.keys()) > 0 and obs0 in self.Q.keys():
            if action in self.Q[obs0].keys():
                ### already been here - no problem
                q_sa = self.Q[obs0][action]
                max_val = max(self.Q[obs1].values()) if obs1 in self.Q.keys() else 0
            else:
                ## been to obs0 but not to action - need to add action as key
                self.Q[obs0][action] = 0
        else:
            ## first time to obs0 - need to create dict with action
            self.Q[obs0] = {action: 0}
        q_target = reward + self.gamma*max_val
        q_delta = q_target - q_sa
        q_sa += q_delta*self.alpha
        self.Q[obs0][action] = q_sa



    def can_reset(self, obs):
        if len(obs["packages"]) == 0:
            return True
        return False


    def can_deliver(self, obs):
        drone_loc = obs["drone_location"]
        target_loc = obs["target_location"]
        packages = [p for p, loc in obs["packages"] if loc=='drone']
        if drone_loc == target_loc:
            if packages:
                return True

        return False


    def can_pickup(self, obs):

        drone_loc = obs["drone_location"]
        packages = [p for p, loc in obs["packages"] if loc=='drone']
        if len(packages)>=2:
            return False
        packages_at_location = [loc for p, loc in obs["packages"] if loc == drone_loc]
        if packages_at_location:
            return True
        return False

    def can_move(self, current_loc):
        """
        Return all locations that drone can move to
        :param p_map: as in init
        :param current_loc: (3, 3)
        :return: list of possible moves for the drone
        """
        i, j = current_loc[0], current_loc[1]
        locations = {'move_up': (i - 1, j), 'move_down': (i + 1, j), 'move_left': (i, j - 1), 'move_right': (i, j + 1)}
        m, n = self.map_shape[1], self.map_shape[0]
        remove = []
        keep = []
        for k, l in locations.items():
            if l[0] < 0 or l[0] >= n or l[1] < 0 or l[1] >= m:
                remove.append(l)
            else:
                keep.append(k)
        return keep


