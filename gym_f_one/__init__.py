from gym.envs.registration import register

register(
    id='f_one-v0',
    entry_point='gym_f_one.envs:FOneEnv',
)