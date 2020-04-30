from src.rl_st import DQNAgent
from src.rl_st import QTableAgent
from src.environment import FurmulaOne
from sandbox.game_pad_inputs import GamePad
from src.lidar import Lidar
import numpy as np
import time
import gym
import sys




def main_q_table():
    env = gym.make('MountainCar-v0')
    q_tab = QTableAgent(num_actions=env.action_space.n, observation_space_low=env.observation_space.low, observation_space_high=env.observation_space.high)


    EPISODES = 10000
    SHOW_EVERY = 500
    MIN_EPSILON = 0.001
    STATS_EVERY = EPISODES // 100
    START_EPSILON_DECAY = 1
    STOP_EPSILON_DECAY = EPISODES // 2
    epsilon = 1
    epsilon_decay_value = epsilon/(STOP_EPSILON_DECAY - START_EPSILON_DECAY)

    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'ave': [], 'max': [], 'min': []}
    for episode in range(EPISODES):
        episode_reward = 0
        discrete_state = q_tab.get_discrete_state(env.reset())
        done = False

        if episode % SHOW_EVERY == 0:
            render = True
            print(episode)
        else:
            render = False

        while not done:

            if np.random.rand() > epsilon:
                action = np.argmax(q_tab.table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            new_discrete_state = q_tab.get_discrete_state(new_state)

            if render:
                env.render()

            if not done:
                q_tab.update_q_table(discrete_state, new_discrete_state, action, reward)

            elif new_state[0] >= env.goal_position:
                q_tab.table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        ep_rewards.append(episode_reward)
        if not episode % STATS_EVERY:
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['ave'].append(sum(ep_rewards[-STATS_EVERY:]) / len(ep_rewards[-STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
            print(f"Episode: {episode}, Epsilon: {epsilon}, ave_reward: {aggr_ep_rewards['ave'][-1]}, min_reward: {aggr_ep_rewards['min'][-1]}, max_reward: {aggr_ep_rewards['max'][-1]}")

        if START_EPSILON_DECAY <= episode <= STOP_EPSILON_DECAY:
            epsilon -= epsilon_decay_value
            epsilon = max(0, epsilon)

    env.close()

    import matplotlib.pyplot as plt

    plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['ave'], label='Ave')
    plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'], label='Min')
    plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'], label='Max')
    plt.legend(loc='upper left')
    plt.show()


def normalise_state(high, low, state):
    return (state-low) / (high-low)

def main_dqn():
    EPISODES = 1000
    START_EPSILON_DECAY = 1
    STOP_EPSILON_DECAY = EPISODES // 1
    epsilon = 0.5
    epsilon_decay_value = 0
    #epsilon_decay_value = epsilon/(STOP_EPSILON_DECAY - START_EPSILON_DECAY)
    MIN_EPSILON = 0.001
    MIN_REWARD = 0.0
    AGGREGATE_STATES_EVERY = EPISODES // 100
    SAVE_EVERY = 10
    SHOW_PREVIEW = False
    TRACK_CAT = 0

    gp = GamePad() # control renders and
    manual_render = False


    env = FurmulaOne()
    #env = gym.make('MountainCar-v0')

    agent = DQNAgent(num_inputs=len(env.observation_space.high), num_actions=env.action_space.n)
    ep_rewards = []

    try:
        for episode in range(EPISODES):
            if agent.tensorboard:
                agent.tensorboard.step = episode

            # Reset the episode params
            episode_reward = 0
            step = 1

            # reset the game and get the initial state
            if env.track.is_store:
                track_num = np.random.randint(TRACK_CAT * env.track.data.cat_length, (TRACK_CAT + 1) * env.track.data.cat_length)
                current_state = env.reset(True, track_num)
            else:
                current_state = env.reset()

            #current_state = normalise_state(env.observation_space.high, env.observation_space.low, current_state)

            # Reset the flag and start iterating until the episode ends
            done = False

            if gp.render_requested:
                manual_render = True
                gp.render_requested = False
            else:
                manual_render = False

            if gp.quit_requested:
                break

            while not done:

                if np.random.random() > epsilon:
                    # Get the action from the Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get a random action
                    action = np.random.randint(0, env.action_space.n)

                new_state, reward, done, _ = env.step(action)

                episode_reward += reward

                if (SHOW_PREVIEW and not episode % AGGREGATE_STATES_EVERY) or manual_render:
                    env.render()

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done, step)

                current_state = new_state
                step += 1

            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATES_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATES_EVERY:]) / len(ep_rewards[-AGGREGATE_STATES_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATES_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATES_EVERY:])

                if agent.tensorboard:
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

                if episode % SAVE_EVERY == 0 and episode != 0:
                    agent.model.save(agent.save_path+f'{agent.name}_{max_reward:_>7.2f}max__{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            print(f'Episode: {episode}, Epsilon {epsilon:.3f}, Reward: {episode_reward:.3f}')
            #print(f'Episode: {episode} Reward: {episode_reward} rLapProgress: {env.r_progress}')
            if START_EPSILON_DECAY <= episode <= STOP_EPSILON_DECAY:
                epsilon -= epsilon_decay_value
                epsilon = max(0, epsilon)

    except KeyboardInterrupt:
        pass
    env.close()
    #gp.exit_thread()


if __name__ == "__main__":
    #main_q_table()
    main_dqn()
