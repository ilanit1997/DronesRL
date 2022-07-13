import search
import numpy as np
import itertools
import json
from copy import deepcopy


class DroneProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node
        --------
        example:
        {
            "map": [['P', 'P', 'P', 'P'],
                    ['P', 'P', 'P', 'P'],
                    ['P', 'I', 'P', 'P'],
                    ['P', 'P', 'P', 'P'], ],
            "drones": {'drone 1': [(3, 3), ['package 1', 'package 2']]},
            "packages": {'package 1': (0, 2),
                         'package 2': (2, 0)},
            "num_packages": 2,
            "clients": {'Yossi': {"path": [(0, 1), (1, 1), (1, 0), (0, 0)],
                                  "packages": ('package 1', 'package 2')
                                  }}
        }
        """

        initial['drones'] = {k: [v, []] for k,v in initial['drones'].items()}
        for c in initial['clients'].keys():
            initial['clients'][c]['current_loc'] = initial['clients'][c]['path'][0]
        initial['num_packages'] = len(initial["packages"])
        self.free_packages = deepcopy(initial['packages'])
        self.packages_clients = {p: 0 for p in initial['packages'].keys()}
        for c in initial['clients'].keys():
            for p in self.packages_clients.keys():
                if p in initial['clients'][c]['packages']:
                    self.packages_clients[p] = c

        hash_initial = json.dumps(initial)
        search.Problem.__init__(self, hash_initial)


    def is_solvable(self, map, packages):
        """
        Checks if any package located on I , if so there is no solution to the problem
        :param map
        :param packages
        :return: False -  not solvable, otherwise True
        """
        for p, p_loc in packages.items():
            if map[p_loc[0]][p_loc[1]] == 'I':
                return False
        return True


    def actions(self, hash_state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        state = json.loads(hash_state)

        if self.is_solvable(state['map'], state['packages']) == False:
            return []

        actions = {d: [] for d in state["drones"].keys()}
        p_map = state["map"]
        for d in actions.keys():
            current_loc, drones_packages = state['drones'][d][0], state['drones'][d][1]
            packages_locations, clients = state['packages'], state['clients']
            drones_loc = state['drones'][d][0]

            #wait action
            if self.can_wait(drones_packages, state, drones_loc):
                actions[d].append(("wait", d))

            #move action
            moves = self.can_move(p_map, current_loc)
            for m in moves:
                actions[d].append(("move", d, m))

            #pickup action
            is_pickup = self.can_pickup(current_loc, drones_packages, packages_locations)
            if is_pickup[0] == True:
                packages_to_pickup = is_pickup[1]
                for p in packages_to_pickup:
                    actions[d].append(("pick up", d, p))

            #delivery action
            is_delivery = self.can_deliver(current_loc, clients, drones_packages)
            if is_delivery[0] == True:
                package_to_deliver = is_delivery[1]
                ##package, client that can be delivered
                for pc in package_to_deliver:
                    p, c = pc[0], pc[1]
                    actions[d].append(("deliver", d, c, p))


        #create_permutations
        allNames = sorted(actions)
        actions = list(itertools.product(*(actions[Name] for Name in allNames)))

        ##remove impossible actions
        actions = self.remove_impossible(actions)

        return actions


    def remove_impossible(self, actions):
        """
        Remove impossible actions such as action which two drones pickup the same package
        :param actions:
        :return: actions list without impossible actions
        """
        to_remove = []
        for action in actions:
            picked_up = []
            for d_action in action:
                if d_action[0] == 'pick up':
                    package = d_action[2]
                    if len(picked_up) == 0:
                        picked_up.append(package)
                    elif package in picked_up:
                        to_remove.append(action)
                    else:
                        picked_up.append(package)
        actions = [x for x in actions if x not in to_remove]
        return actions


    def can_move(self, p_map, current_loc):
        """
        Return all locations that drone can move to
        :param p_map: as in init
        :param current_loc: (3, 3)
        :return: list of possible moves for the drone
        """
        i , j = current_loc[0], current_loc[1]
        locations = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        m , n = len(p_map), len(p_map[0])
        remove = []
        for l in locations:
            if l[0] < 0 or l[0] >= m or l[1] < 0 or l[1] >= n or p_map[l[0]][l[1]] == 'I':
                remove.append(l)
        locations = [x for x in locations if x not in remove]
        return locations


    def can_pickup(self, current_loc, drones_packages, packages):
        """
        Check if drone can pickup a package
        :param drone: 'drone 1'
        :param map: as in init
        :param current_loc: (3, 3)
        :param drones_packages: ['package 1', 'package 2']
        :param packages: {'package 1': (0, 2),
                         'package 2': (2, 0)}
        :return: [True, package_name] if pickup is possible else [False, 0]
        """
        packages_to_pickup = []
        if len(drones_packages) >=2:
            return [False, 0]
        for p, p_loc in packages.items():
            if current_loc == p_loc:
                packages_to_pickup.append(p)

        if len(packages_to_pickup) > 0:
            return [True, packages_to_pickup]
        else:
            return [False, 0]


    def can_deliver(self, drone_loc, clients, drones_packages):
        """"
        Check if drone can deliver a package
        :param drone_loc: (1,2)
        :param clients:   {"clients": {'Yossi': {"path": [(0, 1), (1, 1), (1, 0), (0, 0)],
                                                 "packages": ('package 1', 'package 2')
                                                  }}
        :param drones_packages: ['package 1', 'package 2']
        :return: [True, [packages_name, client_name]] if delivery is possible else [False]
        """
        clients_names = list(clients.keys())
        packages_to_deliver = []
        for p in drones_packages:
            for c in clients_names:
                client_loc = clients[c]['path'][0]
                client_packages = clients[c]['packages']
                if client_loc == drone_loc and p in client_packages:
                    packages_to_deliver.append([p,c])

        if len(packages_to_deliver) > 0:
            return [True, packages_to_deliver]
        else:
            return [False, 0]


    def can_wait(self, drone_packages, state, drones_loc):
        """
        Checks if drone should wait at his location - allow wait only if drones located
        on client's path with his package
        :param drone_packages:
        :param state:
        :param drones_loc:
        :return: False if shouldn't wait, True otherwise
        """
        should_wait = False
        ## drone has some package
        if len(drone_packages) > 0:
            ##go to drones packages clients
            for p in drone_packages:
                client = self.packages_clients[p]
                clients_path = state['clients'][client]['path']
                ## check if drones location is at client path and should wait
                if drones_loc in clients_path:
                    should_wait = True

        if should_wait == False:
            return False

        return True


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of self.actions(state).

        client current loc - ALWAYS CHANGES
        actions:
            move- change drone location
            wait - nothing
            pickup - put in drone's packages
                    drop package from packages
            deliver - num_packages -1
                    drop package - from : 1. drone
                                          2. client
        """
        new_state = json.loads(state)
        clients = new_state['clients'].keys()
        for a in action:
            act = a[0]
            drone = a[1]
            if act == 'move':
                new_state["drones"][drone][0] = a[2]
            if act == 'pick up':
                package = a[2]
                new_state["drones"][drone][1].append(package)
                new_state["packages"] = {k: v for k, v in new_state["packages"].items() if k != package}
            if act == 'deliver':
                package = a[3]
                client = a[2]
                new_state['num_packages'] -=1
                current_packs_drone = new_state["drones"][drone][1]
                new_state["drones"][drone][1] = [p for p in current_packs_drone if p != package]
                current_packs_client = list(new_state["clients"][client]['packages'])
                new_state["clients"][client]['packages'] = tuple([p for p in current_packs_client if p != package])
        ##rotate clients' new path
        for c in clients:
            path = new_state['clients'][c]['path']
            new_state['clients'][c]['path'] = self.rotate(path)

        hash_new_state = json.dumps(new_state)
        return hash_new_state


    def rotate(self, a):
        """ Rotate client's path by one location
         example: [[0,1], [1,1], [1,2]] ==> [[1,1], [1,2], [0,1]]
        param: a - client's path
        return: rotated path """
        len_a = len(a)
        return a[1 % len_a:] + a[:1 % len_a]


    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        hash_state = json.loads(state)

        if hash_state['num_packages'] == 0:
            return True
        return False


    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate
        formula: |P|**2 +
                |P|/(1+len(drones_packages)) +
                min(|P|*min pi \in P dist(drone, pi), dist(drone_{p_i}, client_{p_i}) + my_loc)
        where:
            my_loc : is the # of steps the client has left to do
                     in order to get to drone's location
                     (if drone is at client's path with a package to deliver)
            P : number packages left to deliver
            dist(drone, pi) : defined for every drone
            dist(drone_{p_i}, client_{p_i}): defined if drone has picked up package p_i
                                            which belongs to client_{p_i}

        """

        state = json.loads(node.state)
        drones, packages, p_left = state["drones"], state["packages"], state["num_packages"]
        res = p_left**2

        ## check dead ends
        if node.action == None or p_left == 0:
            return 0
        if node != self.initial:
            if node.state == node.parent.state:
                return float('inf')
        else:
            return 1

        if self.check_loop(node)==True:
            return float('inf')

        my_loc = 0
        ##no more packages left to pick up
        if len(packages.keys()) == 0:
            for d in drones.keys():
                drones_packages, drones_loc = drones[d][1], drones[d][0]
                if len(drones_packages) > 0:
                    item = self.closest_client(drones_packages, drones_loc, state)
                    client, dist_client = item[0], item[1]
                    ## wait or move - check which is better (when client will be at my loc)
                    if dist_client == 0:
                        my_loc = state['clients'][client]['path'].index(drones_loc)

                    res += dist_client + my_loc

        ## some packages left - need to compute what is better : deliver package or pick up package
        else:
            self.distances_dp = self.build_dist_dp(drones, packages)
            dp_allocated = {}
            for dp, dist_package in self.distances_dp.items():
                d, p = dp[0], dp[1]
                drones_packages = drones[d][1]
                dist_client = float('inf')

                #  didn't allocate drone or package
                if self.can_allocate(d, p, dp_allocated, drones_packages):
                    drones_loc = drones[d][0]
                    dp_allocated[d] = p ## mark package - drone allocated
                    res += (p_left) / (len(drones_packages) + 1)

                    #compute closest client
                    if len(drones_packages) > 0:
                        item = self.closest_client(drones_packages, drones_loc, state)
                        client, dist_client = item[0], item[1]
                        ## wait or move - check which is better (when client will be at my loc)
                        if dist_client == 0:
                            my_loc = state['clients'][client]['path'].index(drones_loc)

                #check which is better -  deliver package or pick up package
                res += min(dist_client + my_loc, p_left*dist_package)

        return res


    def build_dist_dp(self, drones, packages):
        """
        calculate distance between drones and packages and then sort by dist
        @return: {(drone 1, package_1): dist(drone 1, package_1), ....} sorted asc by dist
        """
        dp = {}
        for d in drones.keys():
            dic = {tuple([d, p]): self.manhattan_distance(drones[d][0], packages[p]) for p in packages.keys()}
            dp.update(dic)
        dp = dict(sorted(dp.items(), key=lambda item: item[1]))
        return dp

    def can_allocate(self, drone, package, dp_allocated, drones_packages):
        """
        check if drone can assign drone to package
        we go through the list asc by dist, thus we first assign the closest package, drone
        :param drone:
        :param package:
        :param dp_allocated:
        :param drones_packages:
        :return: True if can allocate package to drone, false otherwise
        """
        if len(drones_packages) >=2 :
            return False

        if drone in dp_allocated.keys():
            return False

        if package in dp_allocated.values():
            return False

        return True


    def closest_client(self, drones_packages, drones_loc, state):
        """
        :return closest client, dist to closest loc in clients path
        """
        if len(drones_packages) > 0:
            dist = [(self.packages_clients[p], self.dist_to_client(drones_loc, state['clients'][self.packages_clients[p]]['path']))
                    for p in drones_packages]
            return min(dist, key=lambda x: x[1])

        return float('inf')


    def dist_to_client(self, current_loc, clients_path):
        """
        :return l1 distance between current loc and closest loc in clients path
        """
        clients_path = np.array(clients_path)
        dist = [self.manhattan_distance(current_loc, l) for l in clients_path]
        return min(dist)


    def manhattan_distance(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.abs(a - b).sum()


    def check_loop(self, node):
        """
        :return True if exists a drones that entered a loop, False otherwise
        """
        state = json.loads(node.state)
        actions = node.action
        drones = state["drones"]
        ##example: {drone_1 : [False, (3,3), 1]}
        flags = {d: [True, v[0], i] for i, d, v in zip(range(len(drones.keys())),drones.keys(), drones.values())
                 if actions[i][0] == "move"}

        node = node.parent
        while True:
            proceed = False
            p_state = json.loads(node.state)
            for d in flags.keys():
                i = flags[d][2]
                action = actions[i][0]
                if flags[d][0] == True and action == "move":
                    proceed = True
                    p_loc = p_state["drones"][d][0]
                    if p_loc == flags[d][1]:
                        return True
                else:
                    flags[d][0] = False

            if proceed == False:
                return False

            actions = node.action
            node = node.parent

            if node == None:
                return False


def create_drone_problem(game):
    return DroneProblem(game)


