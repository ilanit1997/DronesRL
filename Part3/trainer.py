import logging
from time import perf_counter

AGENT_ACTION_CALL_TIME_LIMIT = .2
MAXIMUM_STEPS_PER_EPISODE = 30


class DroneTrainer:
    def __init__(self, drone_agent, drone_env):
        self.drone = drone_agent
        self.env = drone_env

    def run(self, nr_episodes: int = 10000, train: bool = False):
        logging.debug(f'training initiated. nr_episodes: {nr_episodes}, mode: {"train" if train else "eval"}')
        if train:
            self.drone.train()
        else:
            self.drone.eval()
        obs0 = self.env.reset()
        steps_counter = 0
        episodes_counter = 0
        reward_list = []
        total_rewards = 0
        done = True
        while episodes_counter < nr_episodes:
            if done:
                logging.debug(f'INFO: episode {episodes_counter} start', end='')
            action = self.drone.select_action(obs0)  # decide on next action
            obs1, reward, done = self.env.step(action)
            self.drone.update(obs0, action, obs1, reward)
            obs0 = obs1
            reward_list.append(reward)
            steps_counter += 1
            done = done or steps_counter >= MAXIMUM_STEPS_PER_EPISODE
            if done:
                logging.debug(f', reward: {sum(reward_list)}')
                total_rewards += sum(reward_list)
                episodes_counter += 1
                steps_counter = 0
                reward_list = []
        return total_rewards/episodes_counter
