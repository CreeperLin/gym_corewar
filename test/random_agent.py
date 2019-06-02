import argparse
import sys

import gym
import gym_corewar
from gym import wrappers, logger

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CoreWar-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id,
        std='icws_88', 
        # act_type='direct_discrete',
        # act_type='direct_continuous',
        # act_type='direct_hybrid',
        # act_type='prog_discrete',
        act_type='prog_continuous',
        # act_type='prog_hybrid',
        coresize=1024,
        maxprocesses=512,
        maxcycles=2000,
        maxsteps=1000,
        dumpintv=4,
        mindistance=20,
        maxlength=20,
        opponents=(
          'warriors/88/simplified/Imp.red',
        #   'warriors/88/simplified/Dwarf.red',
        #   'warriors/88/simplified/MaxProcess.red'
        ),
        initwarrior='warriors/88/simplified/Imp.red',
        recordvideo=True
    )

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/corewar'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 20
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        step = 0
        while True:
            print ('step: %d' % step)
            step+=1
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
