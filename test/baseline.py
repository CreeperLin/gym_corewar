import sys

import gym
import gym_corewar
from gym import wrappers, logger
import baselines
from baselines.run import main

if __name__ == '__main__':

    env_id = 'CoreWar-v0'

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    # env = gym.make(env_id,
    #     std='icws_88', 
    #     # act_type='direct',
    #     act_type='progressive',
    #     coresize=8000,
    #     maxprocesses=8000,
    #     maxcycles=5000,
    #     maxrounds=1000,
    #     dumpintv=4,
    #     mindistance=100,
    #     maxlength=100,
    #     opponents=(
    #       'warriors/88/simplified/Imp.red',
    #       'warriors/88/simplified/Dwarf.red',
    #       'warriors/88/simplified/MaxProcess.red'
    #     ),
    #     # initwarrior='warriors/88/simplified/Imp.red',
    # )

    # # You provide the directory to write to (can be an existing
    # # directory, including one with existing data -- all monitor files
    # # will be namespaced). You can also dump to a tempdir if you'd
    # # like: tempfile.mkdtemp().
    # outdir = '/tmp/corewar'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env.seed(0)

    alg = 'acktr'
    main(['--alg='+alg,'--env='+env_id, '--num_timesteps='+'1e6'])

