import random

from ex2 import DroneAgent, ids
from inputs import small_inputs
import logging
import time
from copy import deepcopy

RESET_PENALTY = 15
DELIVERY_PRICE = 10
TIME_LIMIT = 5



def initiate_agent(state):
    return DroneAgent(state)


class EndOfGame(Exception):
    pass


class DroneStochasticProblem:
    def __init__(self, an_input):
        self.state = an_input
        self.initial_state = deepcopy(self.state)
        start = time.perf_counter()
        self.agent = initiate_agent(self.state)
        end = time.perf_counter()
        if end - start > TIME_LIMIT:
            logging.critical("timed out on constructor")
            raise TimeoutError
        self.score = 0

    def run_round(self):
        while self.state["turns to go"]:
            start = time.perf_counter()
            action = self.agent.act(self.state)
            end = time.perf_counter()
            if end - start > TIME_LIMIT:
                logging.critical(f"timed out on an action")
                print(f'time: {end - start}')
                print(f'time to go: {self.state["turns to go"]}')
                raise TimeoutError
            if not self.is_action_legal(action):
                logging.critical(f"You returned an illegal action!")
                raise RuntimeError
            self.result(action)
        self.terminate_execution()

    def is_action_legal(self, action):
        if action == "reset":
            return True
        if action == "terminate":
            return True
        if len(action) != len(self.state["drones"]):
            logging.error(f"You had given {len(action)} atomic commands, while there are {len(self.state['drones'])}"
                          f" drones in the problem!")
            return False
        drones_already_moved = set()
        for atomic_action in action:
            if not self.is_atomic_action_legal(atomic_action, drones_already_moved):
                logging.error(f"Atomic action {atomic_action} is illegal!")
                return False
        return True

    def is_atomic_action_legal(self, atomic_action, drones_already_moved):
        try:
            action_name = atomic_action[0]
            drone_name = atomic_action[1]
        except TypeError:
            logging.error(f"Your atomic action {atomic_action} has wrong syntax!")
            return False

        if drone_name not in self.state["drones"]:
            logging.error(f"Drone {drone_name} does not exist!")
            return False

        if drone_name in drones_already_moved:
            logging.error(f"Drone {drone_name} was already given command on this turn!")
            return False
        drones_already_moved.add(drone_name)

        if action_name == "wait":
            if len(atomic_action) != 2:
                logging.error(f"Your atomic action {atomic_action} has a wrong syntax!")
                return False
            return True

        if action_name == "pick up":
            if len(atomic_action) != 3:
                logging.error(f"Your atomic action {atomic_action} has a wrong syntax!")
                return False
            package = atomic_action[2]
            if self.state["drones"][drone_name] != self.state["packages"][package]:
                logging.error(f"{drone_name} is not in the same tile as {package}!")
                return False
            return True

        if action_name == "move":
            if len(atomic_action) != 3:
                logging.error(f"Your atomic action {atomic_action} has a wrong syntax!")
                return False
            try:
                origin = self.state["drones"][drone_name]
                destination = atomic_action[2]
                if destination[0] < 0 or destination[1] < 0 or destination[0] >= len(self.state["map"]) or destination[1] >= len(self.state["map"][0]):
                    logging.error(f"You are trying to move to {destination}, which is outside of the grid bounds!")
                    return False
                if self.state["map"][destination[0]][destination[1]] == "I":
                    logging.error(f"You are trying to move to {destination}, which is impassable for drones!")
                    return False
                if abs(origin[0] - destination[0]) > 1 or abs(origin[1] - destination[1]) > 1:
                    logging.error(f"You are trying to move from {origin} to {destination}, which is too far!")
                    return False
            except TypeError:
                logging.error(f"Your atomic action {atomic_action} has a wrong syntax!")
                return False
            return True

        if action_name == "deliver":
            if len(atomic_action) != 4:
                logging.error(f"Your atomic action {atomic_action} has a wrong syntax!")
                return False
            client = atomic_action[2]
            package = atomic_action[3]
            if (self.state["drones"][drone_name] != self.state["clients"][client]["location"]) or \
                    (self.state["packages"][package] != drone_name):
                logging.error(f"{drone_name}, {client}, and {package} are not in the same place!")
                return False
            return True

        return False

    def result(self, action):
        self.apply(action)
        if action != "reset":
            self.environment_step()

    def apply(self, action):
        if action == "reset":
            self.reset_environment()
            return
        if action == "terminate":
            self.terminate_execution()
        for atomic_action in action:
            self.apply_atomic_action(atomic_action)

    def apply_atomic_action(self, atomic_action):
        action_name = atomic_action[0]
        drone_name = atomic_action[1]

        if action_name == "wait":
            return

        if action_name == "pick up":
            package = atomic_action[2]
            self.state["packages"][package] = drone_name

        if action_name == "move":
            destination = atomic_action[2]
            self.state["drones"][drone_name] = destination

        if action_name == "deliver":
            package = atomic_action[3]
            self.state["packages"].pop(package)
            self.score += DELIVERY_PRICE

    def environment_step(self):
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        for client, properties in self.state["clients"].items():
            for _ in range(1000):
                movement = random.choices(movements, weights=properties["probabilities"])[0]
                new_coordinates = (properties["location"][0] + movement[0], properties["location"][1] + movement[1])
                if new_coordinates[0] < 0 or new_coordinates[1] < 0 or new_coordinates[0] >= len(self.state["map"])\
                        or new_coordinates[1] >= len(self.state["map"][0]):
                    continue
                break
            else:
                new_coordinates = (properties["location"][0], properties["location"][1])
            assert new_coordinates
            properties["location"] = new_coordinates

        self.state["turns to go"] -= 1

    def reset_environment(self):
        self.state["packages"] = deepcopy(self.initial_state["packages"])
        self.state["clients"] = deepcopy(self.initial_state["clients"])
        self.state["drones"] = deepcopy(self.initial_state["drones"])
        self.score -= RESET_PENALTY
        self.state["turns to go"] -= 1

    def terminate_execution(self):
        print(f"End of game, your score is {self.score}!")
        print(f"-----------------------------------")
        raise EndOfGame



def main():
    print(f"IDS: {ids}")
    d = {}
    epochs = 1
    for i in range(1, len(small_inputs) + 1):
        score = []
        if i == len(small_inputs):
            epochs = 1
        for _ in range(epochs):
            try:
                inp = deepcopy(small_inputs[i-1])
                my_problem = DroneStochasticProblem(inp)
                my_problem.run_round()
            except EndOfGame:
                score.append(my_problem.score)
                continue
        # d[f'game {i}'] = f'avg score: {sum(score)/epochs}, min score: {min(score)}, max score: {max(score)}'
        # print(f'game {i}', d[f'game {i}'])

if __name__ == '__main__':
    main()

