from gym.envs.registration import register

register(
    id='CoreWar-v0',
    entry_point='gym_corewar.envs:CoreWarEnv',
    # max_episode_steps=200,
    # reward_threshold=195.0,
)