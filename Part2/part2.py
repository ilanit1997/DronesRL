import copy

import numpy as np
import itertools

class DroneAgent:
    """"map": [['P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P'],
            ['P', 'I', 'P', 'P'],
            ['P', 'P', 'P', 'P'], ],
    "drones": {'drone 1': (3, 3)},
    "packages": {'package 1': (2, 2),
                 'package 2': (1, 1)},
    "clients": {'Alice': {"location": (0, 1),
                          "packages": ('package 1', 'package 2'),
                          "probabilities": (0.6, 0.1, 0.1, 0.1, 0.1)}},
    "turns to go": 100"""
    def __init__(self, initial):
        self.delivery_time = 0
        self.time = 0
        self.first_deliver_flag = False
        self.max_delivery_time = float('inf')
        self.num_pack = len(initial["packages"])
        self.packages_clients = {p: 0 for p in initial['packages'].keys()}

        for c in initial['clients'].keys():
            for p in self.packages_clients.keys():
                if p in initial['clients'][c]['packages']:
                    self.packages_clients[p] = c

        ## ILANIT - define distance matrix
        self.cord_dict, self.all_distance = self.create_distances(initial)

        self.prob_mats_t1 = {c : self.build_prob_matrix(initial['clients'][c], initial['map']) for c
                              in initial['clients'].keys()}

        self.prob_mats_t2 = {c: p.T @ p for c, p in self.prob_mats_t1.items()}

        self.probs = [self.prob_mats_t1, self.prob_mats_t2]


    def create_distances(self, state):
        ## ILANIT
        m, n = len(state['map']), len(state['map'][0])
        dist_matrix = {'euclidean': np.zeros((m * n, m * n)), 'manhattan': np.zeros((m * n, m * n))}
        cord_dist = {}
        ##define dict that maps each coordinate on the map to number
        for i in range(m):
            for j in range(n):
                cord_dist[(i, j)] = i*n + j

        for ij1, cord1 in cord_dist.items():
            for ij2, cord2 in cord_dist.items():
                if cord1 == cord2 or cord1 > cord2:
                    continue
                vec_diff = np.array(ij1) - np.array(ij2)
                dist_matrix['euclidean'][cord1, cord2] = np.linalg.norm(vec_diff, ord=2)
                dist_matrix['manhattan'][cord1, cord2] = np.linalg.norm(vec_diff, ord=1)
                dist_matrix['euclidean'][cord2, cord1] = dist_matrix['euclidean'][cord1, cord2]
                dist_matrix['manhattan'][cord2, cord1] = dist_matrix['manhattan'][cord1, cord2]
        return cord_dist, dist_matrix





    def build_prob_matrix(self, client_dict, map):
        """
        :param client_dict:
        :param map:
        :return:  probabilty matrix from i,j to all i,j on the map. We map i, j to -> k = i*m+j such that
        if: k1 = i*m+j and k2 = l*m+r then - prob_matrix[k1][k2] = P(get from i,j to l,r)
        """
        prob = client_dict["probabilities"]
        m, n = len(map), len(map[0])
        p = 0
        prob_matrix = np.zeros((m*n, m*n))
        cord = []
        ##define dict that maps each coordinate on the map to number
        for i in range(m):
            for j in range(n):
                cord.append((i, j))
        ##build prob matrix according to coordinates
        for v in cord:
            i, j = v[0], v[1]
            if i == 0 or j == 0 or i == m-1 or j == n-1:
                if i == 0:
                    # can't go up
                    if j == 0:
                        #can't go up and left
                        p = (prob[0]+prob[2])/3
                        prob_matrix[i * n + j][(i + 1) * n + j] = prob[1] + p
                        prob_matrix[i * n + j][i * n + j + 1] = prob[3] + p
                        prob_matrix[i * n + j][i * n + j] = prob[4] + p
                    elif j == n-1:
                        # can't go up and right
                        p = (prob[0]+prob[3])/3
                        prob_matrix[i * n + j][(i + 1)* n + j] = prob[1] + p
                        prob_matrix[i * n + j][i * n + j - 1] = prob[2] + p
                        prob_matrix[i * n + j][i * n + j] = prob[4] + p
                    else:
                        p = prob[0]/4
                        prob_matrix[i * n + j][(i + 1) * n + j] = prob[1] + p
                        prob_matrix[i * n + j][i * n + j - 1] = prob[2] + p
                        prob_matrix[i * n + j][i * n + j + 1] = prob[3] + p
                        prob_matrix[i * n + j][i * n + j] = prob[4] + p
                elif i == m-1:
                    # can't go down
                    if j == 0:
                        # can't fo down and left
                        p = (prob[1]+prob[2])/3
                        prob_matrix[i * n + j][(i - 1) * n + j] = prob[0] + p
                        prob_matrix[i * n + j][i * n + j + 1] = prob[3] + p
                        prob_matrix[i * n + j][i * n + j] = prob[4] + p
                    elif j == n-1:
                        # can't fo down and right
                        p = (prob[1]+prob[3])/3
                        prob_matrix[i * n + j][(i - 1) * n + j] = prob[0] + p
                        prob_matrix[i * n + j][i * n + j - 1] = prob[2] + p
                        prob_matrix[i * n + j][i * n + j] = prob[4] + p
                    else:
                        p = prob[0]/4
                        prob_matrix[i * n + j][(i - 1) * n + j] = prob[0] + p
                        prob_matrix[i * n + j][i * n + j - 1] = prob[2] + p
                        prob_matrix[i * n + j][i * n + j + 1] = prob[3] + p
                        prob_matrix[i * n + j][i * n + j] = prob[4] + p
                elif j == 0:
                    p = prob[2] / 4
                    prob_matrix[i * n + j][(i - 1) * n + j] = prob[0] + p
                    prob_matrix[i * n + j][(i + 1) * n + j] = prob[1] + p
                    prob_matrix[i * n + j][i * n + j + 1] = prob[3] + p
                    prob_matrix[i * n + j][i * n + j] = prob[4] + p
                elif j == n-1:
                    p = prob[3] / 4
                    prob_matrix[i * n + j][(i - 1) * n + j] = prob[0] + p
                    prob_matrix[i * n + j][(i + 1) * n + j] = prob[1] + p
                    prob_matrix[i * n + j][i * n + j - 1] = prob[2] + p
                    prob_matrix[i * n + j][i * n + j] = prob[4] + p
            else:
                prob_matrix[i * n + j][(i - 1) * n + j] = prob[0] + p
                prob_matrix[i * n + j][(i + 1) * n + j] = prob[1] + p
                prob_matrix[i * n + j][i * n + j - 1] = prob[2] + p
                prob_matrix[i * n + j][i * n + j + 1] = prob[3] + p
                prob_matrix[i * n + j][i * n + j] = prob[4] + p
        return prob_matrix


    def is_solvable(self, map, packages):
        """
        Checks if any package located on I , if so there is no solution to the problem
        :param map
        :param packages
        :return: False -  not solvable, otherwise True
        """
        for p, p_loc in packages.items():
            if type(p_loc) == tuple:
                if map[p_loc[0]][p_loc[1]] != 'I':
                    return True

        return False


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
        i, j = current_loc[0], current_loc[1]
        locations = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1),
                     (i+1, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1)]
        m, n = len(p_map), len(p_map[0])
        remove = []
        for l in locations:
            if l[0] < 0 or l[0] >= m or l[1] < 0 or l[1] >= n or p_map[l[0]][l[1]] == 'I':
                remove.append(l)
        locations = [x for x in locations if x not in remove]
        return locations


    def can_pickup(self, current_loc, drones_packages, packages, packages_picked):
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
        if len(drones_packages) >= 2:
            return [False, 0]
        for p, p_loc in packages.items():
            if current_loc == p_loc and p not in packages_picked:
                packages_to_pickup.append(p)

        if len(packages_to_pickup) > 0:
            return [True, packages_to_pickup]
        else:
            return [False, 0]


    def can_deliver(self, drone_loc, clients, drones_packages):
        """"
        Check if drone can deliver a package
        :param drone_loc: (1,2)
        :param clients:   {"clients": {'Yossi': {
                                                 "packages": ('package 1', 'package 2')
                                                  }}
        :param drones_packages: ['package 1', 'package 2']
        :return: [True, [packages_name, client_name]] if delivery is possible else [False]
        """
        clients_names = list(clients.keys())
        packages_to_deliver = []
        for p in drones_packages:
            for c in clients_names:
                client_loc = clients[c]['location']
                client_packages = clients[c]['packages']
                if client_loc == drone_loc and p in client_packages:
                    packages_to_deliver.append([p, c])

        if len(packages_to_deliver) > 0:
            return [True, packages_to_deliver]
        else:
            return [False, 0]


    def can_wait(self, drone_packages, clients, drones_loc, n):

        if len(drone_packages) == 0:
            return 0

        ## ILANIT - maybe play with different p values
        p_threshold = 0.2
        for p in drone_packages:
            c = self.packages_clients[p]
            i, j = clients[c]['location'][0], clients[c]['location'][1]
            cord1, cord2 = self.cord_dict[(i, j)], self.cord_dict[drones_loc]
            dist_drone_client = self.all_distance['manhattan'][cord1, cord2]

            # dist_drone_client = self.manhattan_distance((i, j), drones_loc)

            ## if the drone is 1 step away from client
            if dist_drone_client == 1:
                P = self.prob_mats_t1[c]
                curr_p = P[i * n + j][drones_loc[0] * n + drones_loc[1]]
                ## if the client can get to drones loc with probability big enough
                if curr_p > p_threshold:
                    return curr_p

            ## if the drone is 2 steps away from client
            elif dist_drone_client == 2:
                P = self.prob_mats_t2[c]
                curr_p = P[i * n + j][drones_loc[0] * n + drones_loc[1]]
                ## if the client can get to drones loc with probability big enough
                if curr_p > p_threshold:
                    return curr_p


        return 0


    def can_reset(self, state):
        num_packs = len(self.packages_clients.keys())
        if num_packs < 2:
           return False

        enough_time = state["turns to go"] - self.delivery_time * 2
        if self.num_pack == 0 and enough_time > 3:
            return True

        return False


    def can_allocate(self, drone, package, dp_allocated, drones_packages):
        """
        check if can assign drone to package
        we go through the list asc by dist, thus we first assign the closest package, drone
        :param drone:
        :param package:
        :param dp_allocated:
        :param drones_packages:
        :return: True if can allocate package to drone, false otherwise
        """
        if len(drones_packages) >=2:
            return False

        if drone in dp_allocated.keys():
            return False

        if package in dp_allocated.values():
            return False

        return True


    def build_dist_dp(self, drones, packages):
        """
        calculate distance between drones and packages and then sort by dist
        @return: {(drone 1, package_1): dist(drone 1, package_1), ....} sorted asc by dist
        """
        dp = {}
        for d in drones.keys():
            ## ILANIT - change to distance dict - maybe return to norm
            dic = {tuple([d, p]): self.all_distance['euclidean'][self.cord_dict[drones[d]], self.cord_dict[packages[p]]]
                   for p in packages.keys()
                   if type(packages[p]) == tuple}
            dp.update(dic)
        dp = dict(sorted(dp.items(), key=lambda item: item[1]))
        return dp


    def min_dist_dc(self, drone_clients, drones_loc, clients, n, eps):
        dists = []
        for c in drone_clients:
            c_loc = clients[c]["location"]
            for k, p in enumerate(self.prob_mats_t1[c][c_loc[0]* n + c_loc[1]]):
                if p == 0:
                    continue
                j = k%n
                i = (k - j) /n
                p_t2 = self.prob_mats_t2[c][c_loc[0] * n + c_loc[1]][k]
                ## ILANIT - change to cord dict - maybe return to norm
                cord1, cord2 = self.cord_dict[(i, j)], self.cord_dict[drones_loc]
                dist_temp = self.all_distance['euclidean'][cord1, cord2]
                dist = (dist_temp + 1/ (p_t2 + eps))/ (p+eps)
                dists.append(dist)

        return min(dists)


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
        new_state = copy.deepcopy(state)

        for a in action:
            act = a[0]
            drone = a[1]
            if act == 'move':
                new_state["drones"][drone] = a[2]
            if act == 'pick up':
                package = a[2]
                new_state["packages"][package] = drone
            if act == 'deliver':
                package = a[3]
                client = a[2]
                new_state["packages"].pop(package)
                current_packs_client = list(new_state["clients"][client]['packages'])
                new_state["clients"][client]['packages'] = tuple([p for p in current_packs_client if p != package])

        return new_state


    def allocate(self, state):
        drones, packages, clients = state["drones"], state["packages"], state["clients"]
        dp_allocated = {}
        self.distances_dp = self.build_dist_dp(drones, packages)

        for dp, dist_package in self.distances_dp.items():
            d, p = dp[0], dp[1]
            drones_packages = [p for p, drone in state['packages'].items() if str(drone) == d]

            #  didn't allocate drone or package
            if self.can_allocate(d, p, dp_allocated, drones_packages):
                dp_allocated[d] = p

        return dp_allocated

    def act(self, state):
        self.time += 1
        self.num_pack = len(state["packages"])

        if self.can_reset(state):
            self.time = 0
            return 'reset'

        allocations = self.allocate(state)
        all_actions = self.actions(state, allocations)

        if len(all_actions) == 1:
            best_action = all_actions[0]
        else:
            actions_values = []
            for i, action in enumerate(all_actions):
                actions_values.append((self.h_old(state, action, allocations), i))

            best_action_idx = min(actions_values)[1]
            best_action = all_actions[best_action_idx]

        if self.num_pack > len(state["packages"]) and self.first_deliver_flag == False:
            self.delivery_time = self.time
            self.first_deliver_flag = True

        # self.print_map(self.result(state, best_action))
        return best_action


    def actions_old(self, state, allocations):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        actions = {d: [] for d in state["drones"].keys()}
        p_map = copy.deepcopy(state["map"])
        packages_picked = []
        drones_to_ignore = []
        m, n = len(p_map), len(p_map[0])
        for d in actions.keys():
            drones_packages = [p for p,drone in state['packages'].items() if str(drone) == d]
            packages_locations = {p:loc for p,loc in state['packages'].items() if type(loc) == tuple}
            current_loc = state['drones'][d]
            clients = state['clients']
            drones_loc = state['drones'][d]

            #delivery action
            is_delivery = self.can_deliver(current_loc, clients, drones_packages)
            #pickup action
            is_pickup = self.can_pickup(current_loc, drones_packages, packages_locations, packages_picked)
            #move action
            moves = self.can_move(p_map, current_loc)

            ## deliver first
            if is_delivery[0] == True:
                package_to_deliver = is_delivery[1]
                ##package, client that can be delivered
                for pc in package_to_deliver:
                    p, c = pc[0], pc[1]
                    actions[d].append(("deliver", d, c, p))
            #pickup action
            elif is_pickup[0] == True :
                packages_to_pickup = is_pickup[1]
                for p in packages_to_pickup:
                    if len(packages_picked)>0:
                        if p not in packages_picked:
                            actions[d].append(("pick up", d, p))
                            packages_picked.append(p)
                            break
                    else:
                        actions[d].append(("pick up", d, p))
                        packages_picked.append(p)
                        break

            #wait - only if probability is big enought
            elif self.can_wait(drones_packages, clients, drones_loc, n) > 0:
                actions[d].append(("wait", d))

            ##  not holding any package and not allocated package - just wait
            elif len(drones_packages) == 0 and d not in allocations.keys():
                drones_to_ignore.append(d)
                actions[d].append(("wait", d))

            #move action
            elif len(moves) != 0:

                for m1 in moves:
                    actions[d].append(("move", d, m1))

            else:
                actions[d].append(("wait", d))

        #create_permutations
        allNames = sorted(actions)
        actions = list(itertools.product(*(actions[Name] for Name in allNames)))
        ##remove impossible actions
        actions = self.remove_impossible(actions)

        return actions




    # def old_h(self, state):
    #     drones, packages, clients = state["drones"], state["packages"], state["clients"]
    #     p_left = len(packages.keys())
    #     res = p_left**2
    #     m, n = len(state["map"]), len(state["map"][0])
    #     eps = 0.001
    #     WAIT_DISTANCE = 0
    #
    #     if p_left == 0:
    #         return 0
    #     packages_left_topick = [(p,v) for p,v in packages.items() if type(v) == tuple]
    #     ##no more packages left to pick up
    #     if len(packages_left_topick) == 0:
    #         for d in drones.keys():
    #             drones_packages = [p for p, drone in state['packages'].items() if str(drone) == d]
    #             drones_loc = drones[d]
    #             if len(drones_packages) > 0:
    #                 drone_clients = [self.packages_clients[p] for p in drones_packages]
    #                 drone_clients_dist = {c: self.all_distance['manhattan'][self.cord_dict[clients[c]["location"]], self.cord_dict[drones_loc]]
    #                                       for c in drone_clients}
    #                 p_res = 0
    #                 for c in drone_clients_dist.keys():
    #                     ## drone is one step from delivery to client - should wait if p is big enough
    #                     if drone_clients_dist[c] == WAIT_DISTANCE:
    #                         i, j = clients[c]["location"][0], clients[c]["location"][1]
    #                         p_res += (self.prob_mats_t1[c][i * n + j][drones_loc[0]*n + drones_loc[1]] + eps)
    #
    #                     ## drone is far from client - look where should move
    #                     else:
    #                         res += self.min_dist_dc(drone_clients, drones_loc, clients, n, eps)
    #
    #                 res += 1 / (p_res + eps)
    #                 res*= (p_left*4)**2 / (len(drones_packages)*4 + 1)**4
    #                 res += p_left ** 2
    #
    #     else:
    #         self.distances_dp = self.build_dist_dp(drones, packages)
    #         dp_allocated = {}
    #         for dp, dist_package in self.distances_dp.items():
    #             d, p = dp[0], dp[1]
    #             drones_packages = [p for p, drone in state['packages'].items() if str(drone) == d]
    #             drones_loc = drones[d]
    #             dist_client = float('inf')
    #             p_res = eps
    #
    #             #  didn't allocate drone or package
    #             if self.can_allocate(d, p, dp_allocated, drones_packages):
    #                 ## mark package - drone allocated
    #                 dp_allocated[d] = p
    #                 res += (p_left*4 + 1)**2 / (len(drones_packages)*4 + 1)**4
    #
    #
    #                 # compute closest client
    #                 if len(drones_packages) > 0:
    #                     drone_clients = [self.packages_clients[p] for p in drones_packages]
    #                     drone_clients_dist = {c: self.manhattan_distance(clients[c]["location"], drones_loc)
    #                                           for c in drone_clients}
    #                     for c in drone_clients_dist.keys():
    #                         ## drone is one step from delivery to client - should wait if p is big enough
    #                         if drone_clients_dist[c] == WAIT_DISTANCE:
    #                             i, j = clients[c]["location"][0], clients[c]["location"][1]
    #                             p_res += (self.prob_mats_t1[c][i * n + j][drones_loc[0] * n + drones_loc[1]] + eps)
    #
    #                         ## drone is far from client - look where should move
    #                         else:
    #                             dist_client = self.min_dist_dc(drone_clients, drones_loc, clients, n, eps)
    #             res += min(dist_client + 1/(p_res+eps), p_left*dist_package)
    #             res += p_left ** 2
    #
    #     return res

    def actions(self, state, allocations):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        actions = {d: [] for d in state["drones"].keys()}
        p_map = copy.deepcopy(state["map"])
        packages_picked = []
        m, n = len(p_map), len(p_map[0])
        for d in actions.keys():
            drones_packages = [p for p, drone in state['packages'].items() if str(drone) == d]
            packages_locations = {p: loc for p, loc in state['packages'].items() if type(loc) == tuple}
            current_loc = state['drones'][d]
            clients = state['clients']
            drones_loc = state['drones'][d]

            # delivery action
            is_delivery = self.can_deliver(current_loc, clients, drones_packages)
            # pickup action
            is_pickup = self.can_pickup(current_loc, drones_packages, packages_locations, packages_picked)
            # move action
            moves = self.can_move(p_map, current_loc)

            ## deliver first
            if is_delivery[0] == True:
                package_to_deliver = is_delivery[1]
                ##package, client that can be delivered
                for pc in package_to_deliver:
                    p, c = pc[0], pc[1]
                    actions[d].append(("deliver", d, c, p))
            # pickup action
            elif is_pickup[0] == True:
                packages_to_pickup = is_pickup[1]
                for p in packages_to_pickup:
                    if len(packages_picked) > 0:
                        if p not in packages_picked:
                            actions[d].append(("pick up", d, p))
                            packages_picked.append(p)
                            break
                    else:
                        actions[d].append(("pick up", d, p))
                        packages_picked.append(p)
                        break

            # wait - only if probability is big enought
            elif self.can_wait(drones_packages, clients, drones_loc, n) > 0:
                actions[d].append(("wait", d))

            ## ILANIT
            ##  not holding any package and not allocated package - just wait until game is over
            elif len(drones_packages) == 0 and d not in allocations.keys():
                actions[d].append(("wait", d))

            ## ILANIT
            # move action - choose what is best move
            elif len(moves) != 0:
                best_move = self.h(moves, d, state, allocations)
                actions[d].append(("move", d, best_move))

            else:
                actions[d].append(("wait", d))

        # create_permutations
        allNames = sorted(actions)
        actions = list(itertools.product(*(actions[Name] for Name in allNames)))
        ##remove impossible actions
        actions = self.remove_impossible(actions)
        return actions


    def h(self, moves, d, state, allocations):
        drones, packages, clients = state["drones"], state["packages"], state["clients"]
        p_left = len(packages.keys())
        m, n = len(state["map"]), len(state["map"][0])
        if p_left == 0:
            return 0

        ## ILANIT
        moves_scores = {}
        for drones_loc in moves:
            dist_client, dist_package = float('inf'), float('inf')
            drones_packages = [p for p, drone in state['packages'].items() if str(drone) == d]
            drone_clients = [self.packages_clients[p] for p in drones_packages]
            ## meaning we allocated package to drone - already checked if legal
            if len(allocations.keys()) > 0:
                if d in allocations.keys():
                    p_loc = packages[allocations[d]]
                    cord_p, cord_d = self.cord_dict[p_loc], self.cord_dict[drones_loc]
                    dist_package = self.all_distance['euclidean'][cord_p, cord_d]

            # compute closest client if already has package
            if len(drones_packages) > 0:
                dist_client = self.min_dist_dc(drone_clients, drones_loc, clients, n, eps = 0.001)

            ## ILANIT - replaces res
            moves_scores[drones_loc] = min(dist_client, p_left*dist_package)

        ## ILANIT
        best_move = min(moves_scores.items(), key=lambda x: x[1])[0]

        return best_move


    def h_old(self, state, action, allocations):
        drones, packages, clients = state["drones"], state["packages"], state["clients"]
        p_left = len(packages.keys())
        p_drones = 0
        res = 0
        m, n = len(state["map"]), len(state["map"][0])
        if p_left == 0:
            return 0

        for a in action:
            act = a[0]
            d = a[1]
            if act == 'move':
                dist_client, dist_package = float('inf'), float('inf')
                drones_packages = [p for p, drone in state['packages'].items() if str(drone) == d]
                p_drones += len(drones_packages)
                drone_clients = [self.packages_clients[p] for p in drones_packages]
                drones_loc = a[2]
                ## meaning we allocated package to drone - already checked if legal
                if len(allocations.keys()) > 0:
                    if d in allocations.keys():
                        p_loc = packages[allocations[d]]
                        cord_p, cord_d = self.cord_dict[p_loc], self.cord_dict[drones_loc]
                        dist_package = self.all_distance['euclidean'][cord_p, cord_d]

                # compute closest client if already has package
                if len(drones_packages) > 0:
                    dist_client = self.min_dist_dc(drone_clients, drones_loc, clients, n, eps = 0.001)

                res += min(dist_client, p_left*dist_package)

        return res



    def print_map(self, state):
        CRED = '\033[91m'
        CEND = '\033[0m'
        CGREEN = '\33[32m'
        CBLUE = '\33[34m'
        CBLACK = '\33[30m'
        drones, packages, clients = state['drones'],  state['packages'],  state['clients']
        temp_map = state['map']
        cor_dict = {}
        for i in range(len(temp_map)):
            for j in range(len(temp_map[0])):
                cor_dict[(i,j)] = [temp_map[i][j]]

        for d in drones.keys():
            d_loc = drones[d]
            d_num = d.split(' ')[1]
            i, j = d_loc[0], d_loc[1]
            cor_dict[(i,j)].append('#d'+ d_num)
            if d in packages.values():
                try:
                    drones_packages = [p.split(' ')[1] for p, drone in state['packages'].items() if str(drone) == d]
                except Exception:
                    drones_packages = [p for p, drone in state['packages'].items() if str(drone) == d]

                cor_dict[(i,j)].append('#p' + ','.join(drones_packages))

        for c in clients.keys():
            c_loc = clients[c]['location']
            i, j = c_loc[0], c_loc[1]
            cor_dict[(i,j)].append('#c' + c[0])


        for p in packages.keys():
            curr_p = packages[p]
            if type(curr_p) == tuple:
                i, j = curr_p[0], curr_p[1]
                if ' ' in p:
                    p_num = p.split(' ')[1]
                else:
                    p_num = p
                cor_dict[(i,j)].append('#p' + p_num)

        for ij, items in cor_dict.items():
            if len(items) > 1:
                temp_map[ij[0]][ij[1]] = ''.join(items[1:])

        import os
        # System call
        os.system("")
        map = np.array(temp_map)
        for row in range(len(map)):
            print('+' + '-+' * len(map[0]))
            print(f'{row}: |', end='')
            for col in range(len(map[row])):
                curr = map[row][col]
                if '#p' in curr:
                    START = CGREEN
                elif '#d' in curr:
                    START = CBLUE
                elif '#c' in curr:
                    START = CRED
                else:
                    START = CBLACK

                print(START + map[row][col] + CEND, end='|')
            print(' ')  # To change lines
        print('+' + '-+' * (len(map[0])))
