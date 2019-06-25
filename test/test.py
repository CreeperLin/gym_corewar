import gym
import gym_corewar
import numpy as np

last_score = 0
def rfunc(match, cycles, sum_proc):
  global last_score
  c = sum_proc.shape[0]
  for i in range(sum_proc.shape[1]):
    dat = sum_proc[:,i]
    print('w%d: end: %d max: %d avg: %f' % (i, dat[c-1], np.max(dat), np.average(dat)))
  w1 = np.array([(1.2**(2*(i+1)/c))/(1.2**2) for i in range(c)])
  w2 = np.array([-1.0]*2)
  w2[0] = 1.0
  r_proc = 100 * np.dot (np.dot (w1, sum_proc), w2) / (cycles + 1) / 2
  print('score proc %f' % r_proc)
  r_dura = 100 * cycles / 2000
  print('score dura %f' % r_dura)
  r_goal = 1 if match > 0 else 0 if match == 0 else 0.25
  r_goal = 1000 * r_goal * cycles / 200
  print('score goal %f' % r_goal)
  score = r_proc + r_dura + r_goal
  ret = score - last_score + 1000 * (1 if match > 0 else 0) * cycles / 2000
  print('score %f reward %f' % (score, ret))
  last_score = score
  return ret

if __name__=="__main__":
  env = gym.make('CoreWar-v0',
    std='icws_88', 
    # act_type='direct_discrete',
    act_type='direct_continuous',
    # act_type='direct_hybrid',
    # act_type='prog_discrete',
    # act_type='prog_continuous',
    # act_type='prog_hybrid',
    coresize=1024,
    maxprocesses=512,
    maxcycles=2000,
    dumpintv=1,
    mindistance=32,
    maxlength=32,
    opponents=(
      # 'warriors/88/simplified/Imp.red',
      # 'warriors/88/simplified/Dwarf.red',
      # 'warriors/88/simplified/Simple_88.red'
      'warriors/88/simplified/Wait.red'
    ),
    # initwarrior='warriors/88/simplified/Imp.red',
    randomize=True,
    verbose=True,
    recordvideo=True,
    rewardfunc=rfunc
  )
  obs = env.reset()
  steps = 0
  total_reward = 0
  wins = 0
  loses = 0
  ties = 0
  epoch = int(1e7)
  for _ in range(epoch):
    print('step %d' % _)
    # env.render('ansi')
    # env.render('human')
    a = env.action_space.sample()
    obs, r, done, info = env.step(a)
    m = info['match']
    if (m==-1): ties += 1
    elif (m==0): loses += 1
    else: wins += 1
    if steps % 20 == 0 or done:
      print("\n### step {} total_reward {:+0.2f}".format(steps, total_reward))
    steps += 1
    total_reward += r
    if done: break
  env.close()
  print('win %d lose %d tie %d' % (wins, loses, ties))
  for i in range(len(env.winners)):
    print('winner %d' % i)
    for j in range(len(env.winners[i])):
      print(env.winners[i][j])
