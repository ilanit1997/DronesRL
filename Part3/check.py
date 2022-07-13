import logging
from time import perf_counter

from ex3 import DroneAgent, ids
from inputs import inputs_list
from drone_env import DroneEnv
from trainer import DroneTrainer

AGENT_INIT_TIME_LIMIT = 1.
EPISODE_TIME_LIMIT = 4e-3
NR_TRAIN_EPISODES = int(200000)
NR_TEST_EPISODES = int(10e3)

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    logging.info(f"IDS:: {ids}")
    alpha_list = [0.5]
    gamma_list = [ 0.95, 0.95,  0.95,  0.95, 0.95 , 0.95]

    scores = {}
    for alpha in alpha_list:
        for gamma in gamma_list:
            test_scores = []
            for idx, params in enumerate(inputs_list):
                try:
                    logging.info(f'input_id:: {idx}')
                    """ Initialize Environment """
                    drone_env = DroneEnv(params)
                    """ Create Agent """
                    n = len(params['map'])
                    m = len(params['map'][0])
                    start = perf_counter()
                    drone_agent = DroneAgent(n, m, alpha=alpha, gamma=gamma)
                    end = perf_counter()
                    if end - start > AGENT_INIT_TIME_LIMIT:
                        logging.critical(f"timed out on agent constructor, time: {round(end - start, 2)}")
                        raise TimeoutError
                    """ Run Trainer """
                    trainer = DroneTrainer(drone_agent, drone_env)
                    start = perf_counter()
                    average_score_train = trainer.run(nr_episodes=NR_TRAIN_EPISODES, train=True)
                    end = perf_counter()
                    if end - start > EPISODE_TIME_LIMIT * NR_TRAIN_EPISODES:
                        logging.critical(f"timed out on train, time: {round(end - start, 2)}")
                        raise TimeoutError
                    logging.info(f'train score: {average_score_train}, time: {round(end - start, 2)}')
                    """ Evaluate Agent"""
                    start = perf_counter()
                    average_score_test = trainer.run(nr_episodes=NR_TEST_EPISODES, train=False)
                    end = perf_counter()
                    if end - start > EPISODE_TIME_LIMIT * NR_TEST_EPISODES:
                        logging.critical(f"timed out on test, time: {round(end - start, 2)}")
                        raise TimeoutError
                    logging.info(f'test score: {average_score_test}, time: {round(end - start, 2)}')
                    test_scores.append(average_score_test)
                except TimeoutError:
                    test_scores.append(-50.)
                    continue
            logging.info(f"Done!")
            logging.info(f"scores: {test_scores}")
            scores[f'alpha={str(alpha)}, gamma={str(gamma)}'] = test_scores
            print(f'alpha={str(alpha)}, gamma={str(gamma)}:')
            print(test_scores)


    ##print best params:
    arr = {f'{i}': [] for i in range(8)}
    for key, item in scores.items():
        for i in range(8):
            arr[str(i)].append((item[i], key))

    scores_sorted = dict(sorted(arr.items(), key=lambda x: x[1][0]))
    for item in scores_sorted.items():
        print(item)