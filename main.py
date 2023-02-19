import gym
import csv
import numpy as np
import multiprocessing
import time
import gym_taco_environments
from agent import MonteCarloDeterministic, MonteCarloStochastic

deterministic_fieldnames = [
    "iter_num",
    "method",
    "gamma",
    "epsilon",
    "reward",
    "steps",
]

stochastic_fieldnames = [
    "iter_num",
    "method",
    "gamma",
    "epsilon",
    "reward",
    "steps",
]

class agent_deterministic_arguments:
    """
    agent_arguments allows to encapsulate all of the parameters needed to run a
    new episode for our agent and doing concurrently on the system that is
    running it.
    """

    def __init__(self, env, experiments, episodes, method, gamma):
        self.experiments = experiments
        self.episodes = episodes
        self.method = method
        self.gamma = gamma
        self.env = env

    def __str__(self) -> str:
        return f"<{self.episodes}|{self.method}|{self.gamma}>"

class agent_stochastic_arguments:
    """
    agent_arguments allows to encapsulate all of the parameters needed to run a
    new episode for our agent and doing concurrently on the system that is
    running it.
    """

    def __init__(self, env, experiments, episodes, method, gamma, epsilon):
        self.experiments = experiments
        self.episodes = episodes
        self.method = method
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env

    def __str__(self) -> str:
        return f"<{self.episodes}|{self.method}|{self.gamma}|{self.epsilon}>"

def train(env, agent, episodes):
    reward_total = 0.0
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation)
            new_observation, reward, terminated, truncated, _ = env.step(action)
            reward_total += reward
            agent.update(observation, action, reward, terminated)
            observation = new_observation
    return reward_total


def play(env, agent):
    observation, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.get_best_action(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        env.render()
        #time.sleep(1)

# run learning process takes the arguments for the deterministic agent and runs the specified
# amount of experiments also while returning summaries from that experiments set.
def run_deterministic_learning_process(arguments: agent_deterministic_arguments):
    #print(str(arguments))

    average_values = {
        "reward": 0.0,
        "steps": arguments.episodes,
    }

    for _ in range(arguments.experiments):
        agent = MonteCarloDeterministic(
            arguments.env.observation_space.n,
            arguments.env.action_space.n,
            arguments.gamma,
        )
        reward = 0
        reward += train(env, agent, episodes=arguments.episodes)
        #ep_q, ep_pi = agent.render()
        #print('Q', ep_q, 'Pi', ep_pi)
        agent.reset()

        play(env, agent)

    average_values["reward"] /= arguments.experiments
    average_values["steps"] /= arguments.experiments
    average_values["method"] = arguments.method
    average_values["gamma"] = arguments.gamma

    return average_values

# run learning process takes the arguments for the deterministic agent and runs the specified
# amount of experiments also while returning summaries from that experiments set.
def run_stochastic_learning_process(arguments: agent_stochastic_arguments):
    #print(str(arguments))

    average_values = {
        "reward": 0.0,
        "steps": arguments.episodes,
    }

    for _ in range(arguments.experiments):
        agent = MonteCarloStochastic(
            arguments.env.observation_space.n,
            arguments.env.action_space.n,
            arguments.gamma,
            arguments.epsilon,
        )
        reward = 0
        reward += train(env, agent, episodes=arguments.episodes)
        #ep_q, ep_pi = agent.render()
        #print('Q', ep_q, 'Pi', ep_pi)
        agent.reset()

        play(env, agent)

    average_values["reward"] /= arguments.experiments
    average_values["steps"] /= arguments.experiments
    average_values["method"] = arguments.method
    average_values["gamma"] = arguments.gamma
    average_values["epsilon"] = arguments.epsilon

    return average_values


if __name__ == "__main__":
    env = gym.make("FrozenMaze-v0", render_mode="ansi", delay=0.0001)
    
    experiments = 100

    episodes = [10, 100, 1000]
    methods = ["deterministic", "stochastic"]

    deterministic_arguments = []
    stochastic_arguments = []

    for method in methods:
        for episode in episodes:
            for gamma in np.arange(0.1, 1.0, 0.1):

                if method == "deterministic":
                    deterministic_arguments.append(
                            agent_deterministic_arguments(
                                episodes=episode,
                                gamma=gamma,
                                method=method,
                                env=env,
                                experiments=experiments,
                            )
                        )

                for epsilon in np.arange(0.1, 1.0, 0.1):
                    if method == "stochastic":
                        stochastic_arguments.append(
                            agent_stochastic_arguments(
                                episodes=episode,
                                gamma=gamma,
                                epsilon=epsilon,
                                method=method,
                                env=env,
                                experiments=experiments,
                            )
                        )
    f = open("summary_deterministic.csv", "a", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=deterministic_fieldnames)

        for deterministic_argument in deterministic_arguments:
            values = run_deterministic_learning_process(deterministic_argument)

            writer.writerows([values])

    f = open("summary_stochastic.csv", "a", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=stochastic_fieldnames)

        for stochastic_argument in stochastic_arguments:
            values = run_stochastic_learning_process(stochastic_argument)

            writer.writerows([values])
