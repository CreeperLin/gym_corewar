import sys
import os
import numpy as np
import gym
from gym import error, spaces, utils
from gym.spaces import Discrete, Tuple, Box, MultiDiscrete
from gym.utils import seeding
import Corewar
import Corewar.Benchmarking
from Corewar.Redcode import *

def OPCODE88(insn):
	return ((insn) >> 4)

def OPCODE(insn):
  return ((insn) >> 9)

def MODIFIER(insn):
  return (((insn) >> 6) & 0x0007)

def SETMODIFIER(insn,mod):
  return (((insn) & 0x3e3f) | ((mod) << 6))

def AMODE88(insn):
  return (((insn) & 0x0c) >>  2)

def AMODE(insn):
  return (((insn) >> 3) & 0x0007)

def SETAMODE88(insn,am):
  return (((insn) & 0xf3) | ((am) <<  2))

def SETAMODE(insn,am):
  return (((insn) & 0x3fc7) | ((am) << 3))

def BMODE88(insn):
  return ((insn) & 0x03)

def BMODE(insn):
  return ((insn) & 0x0007)

def SETBMODE88(insn,bm):
  return ((insn & 0xfc) | (bm))

def SETBMODE(insn,bm):
  return ((insn & 0x3ff8) | (bm))

valid_opcodes = ('dat', 'mov', 'add', 'sub', 'jmp', 'jmz', 'jmn', 'djn',
                 'cmp', 'spl', 'slt', 'mul', 'div', 'mod', 'seq', 'sne',
                 'nop', 'ldp', 'stp')

OPCODES = (OPCODE_DAT, OPCODE_MOV, OPCODE_ADD,
          OPCODE_SUB, OPCODE_JMP, OPCODE_JMZ,
          OPCODE_JMN, OPCODE_DJN, OPCODE_CMP,
          OPCODE_SPL, OPCODE_SLT, OPCODE_MUL,
          OPCODE_DIV, OPCODE_MOD, OPCODE_SEQ,
          OPCODE_SNE, OPCODE_NOP, OPCODE_LDP,
          OPCODE_STP)

valid_modifiers = ('a', 'b', 'f', 'i', 'x', 'ab', 'ba')
MODIFIERS = (MODIFIER_A, MODIFIER_B, MODIFIER_F,
            MODIFIER_I, MODIFIER_X, MODIFIER_AB,
            MODIFIER_BA)

valid_modes = ('#', '$', '@', '<', '>', '*', '{', '}')
MODES = (MODE_IMMEDIATE, MODE_DIRECT, MODE_B_INDIRECT,
            MODE_B_PREDECREMENT, MODE_B_POSTINCREMENT,
            MODE_A_INDIRECT, MODE_A_PREDECREMENT,
            MODE_A_POSTINCREMENT)

dim_opcode_88 = 11
dim_opcode_94_nop = 17
dim_opcode_94 = 19
dim_modifiers_88 = 0
dim_modifiers_94_nop = 7
dim_modifiers_94 = 7
dim_addrmodes_88 = 4
dim_addrmodes_94_nop = 8
dim_addrmodes_94 = 8
valid_pseudoopcodes = ('org', 'end', 'for', 'rof', 'equ', 'pin')
dim_progress = 3

class CoreWarEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self,
        std='icws_88', 
        act_type='direct',
        obs_type='full',
        coresize=8000,
        maxprocesses=8000,
        maxcycles=10000,
        dumpintv=100,
        mindistance=25,
        maxlength=25,
        pspacesize=None,
        opponents=None,
        numplayers=2,
        ):
    self.viewer = None
    self.core_size = coresize
    self.max_proc = maxprocesses
    self.max_cycle = maxcycles
    self.min_dist = mindistance
    self.max_length = maxlength
    self.obs_dump_intv = dumpintv
    self.num_players = numplayers

    if std=='icws_88':
      self.dim_opcode = dim_opcode_88
      self.dim_modifiers = dim_modifiers_88
      self.dim_addrmodes = dim_addrmodes_88
      self._get_inst = Instruction88
      self._OPCODE = OPCODE88
      self._AMODE = AMODE88
      self._BMODE = BMODE88
      # self._MODIFIER = MODIFIER
      self.parser = Corewar.Parser(coresize=coresize,
                              maxprocesses=maxprocesses,
                              maxcycles=maxcycles,
                              maxlength=maxlength,
                              mindistance=mindistance,
                              standard=Corewar.STANDARD_88)
      self.mars = Corewar.Benchmarking.MARS_88(coresize = coresize,
                              maxprocesses = maxprocesses,
                              maxcycles = maxcycles,
                              mindistance = mindistance,
                              maxlength = maxlength)
    elif std=='icws_94_nop':
      self.dim_opcode = dim_opcode_94_nop
      self.dim_modifiers = dim_modifiers_94_nop
      self.dim_addrmodes = dim_addrmodes_94_nop
      self._get_inst = Instruction
      self._OPCODE = OPCODE
      self._AMODE = AMODE
      self._BMODE = BMODE
      self._MODIFIER = MODIFIER
      self.parser = Corewar.Parser(coresize=coresize,
                              maxprocesses=maxprocesses,
                              maxcycles=maxcycles,
                              maxlength=maxlength,
                              mindistance=mindistance,
                              standard=Corewar.STANDARD_94_NOP)
      self.mars = Corewar.Benchmarking.MARS_94nop(coresize = coresize,
                              maxprocesses = maxprocesses,
                              maxcycles = maxcycles,
                              mindistance = mindistance,
                              maxlength = maxlength)
    elif std=='icws_94':
      self.dim_opcode = dim_opcode_94
      self.dim_modifiers = dim_modifiers_94
      self.dim_addrmodes = dim_addrmodes_94
      self._get_inst = Instruction
      if not pspacesize:
          pspacesize = coresize / 16
      else:
          pspacesize = pspacesize
      self.parser = Corewar.Parser(coresize=coresize,
                              maxprocesses=maxprocesses,
                              maxcycles=maxcycles,
                              maxlength=maxlength,
                              mindistance=mindistance,
                              pspacesize=pspacesize,
                              standard=Corewar.STANDARD_94)
      raise ValueError("standard icws94 not supported")
    else:
      raise ValueError("invalid standard")

    self.opponents = []
    if (opponents == None):
      opponents = ('warriors/88/Imp.red',
      'warriors/88/Dwarf.red',
      'warriors/88/MaxProcess.red')
    for i in range(len(opponents)):
      print('reading warrior in %s' % (opponents[i]))
      self.opponents.append(self.parser.parse_file(opponents[i]))

    self.num_insn = self.dim_opcode*self.dim_addrmodes*self.dim_addrmodes
    if (act_type=='direct'):
      self.dim_action_insn = (self.max_length, )
      self.dim_action_field = (self.max_length, 2)
      self._parse_act = self._parse_act_direct
      self.action_space = Tuple((
                                Box(low=0, high=self.num_insn-1,
                                    shape=self.dim_action_insn, dtype=np.uint16),
                                # Box(low=-self.core_size/2+1, high=self.core_size/2,
                                Box(low=0, high=self.core_size-1,
                                    shape=self.dim_action_field, dtype=np.uint16)))
    elif (act_type=='progressive'):
      self._parse_act = self._parse_act_prog
      self.action_space = Tuple((
                                Discrete(dim_progress),
                                Box(low=np.array([0, 0, 0]), 
                                    high=np.array([self.num_insn-1, self.core_size-1, self.core_size-1]), dtype=np.uint16)))
    else:
      raise ValueError("invalid action space type")
                            
    self.dim_obs_sample = int(self.max_cycle // self.obs_dump_intv)
    if (obs_type == 'full'):
      self._get_obs = self._get_obs_full
      self.dim_obs_insn  = (self.dim_obs_sample, self.core_size, )
      self.dim_obs_field = (self.dim_obs_sample, self.core_size, 2)
      self.observation_space = Tuple((
                                Box(low=0, high=self.num_insn-1, 
                                    shape=self.dim_obs_insn, dtype=np.uint16),
                                # Box(low=-self.core_size/2+1, high=self.core_size/2,
                                Box(low=0, high=self.core_size-1,
                                    shape=self.dim_obs_field, dtype=np.uint16)))
    elif (obs_type == 'warriors'):
      raise ValueError("obs space type not supported")
    else:
      raise ValueError("invalid observation space type")
    
    self.turn = -1

  def _get_image(self):
    return None

  def step(self, action):
    if (self.turn==-1):
      raise ValueError("needs reset")
    
    done = False

    self._parse_act(action)

    # res = self.mars.open((self.opponent, self.opponent), seed = self.seed)
    res = self.mars.open((self.warrior, self.opponent), seed = self.seed)
    
    if (res!=0):
      raise ValueError("error opening MARS")

    print('\n%s fighting with %s (%d)' % (self.warrior.name, self.opponent.name, self.seed))

    match = -1
    obs_cnt = 0
    while (True):
      res = self.mars.step()
      if (res==0):
        print("warrior %d lose" % self.turn)
        match = self.turn
        break
      if self.cycles >= self.max_cycle:
        break
      else:
        # print("warrior %d: %d" % (self.turn, res))
        if (self.cycles % self.num_players == 0):
          tmp = [0] * self.num_players
          tmp[0] = res
          self.sum_proc = np.append(self.sum_proc, [tmp], axis=0)
        else:
          self.sum_proc[int(self.cycles // self.num_players)+1][self.turn] = res

      self.cycles += 1
      self.turn = (self.turn + 1) % self.num_players

      if (self.cycles % self.obs_dump_intv == 0):
        self.coredump[obs_cnt, :] = np.array(self.mars.dumpcore(), dtype=np.uint16)
        obs_cnt+=1
    
    self.mars.stop()
    print('cycle: %d' % (self.cycles))

    if (match>0):
      print('winner!')
      print(self.warrior)
      self.winners.append(self.warrior.instructions.copy())

    s = self._get_obs()
    r = self._get_reward()
    self._reset()
    return s, r, done, {'match':match}

  def _parse_act_prog(self, action):
    raise ValueError("not supported")

  def _parse_act_direct(self, action):
    _insn, _field = action
    clen = _insn.shape[0]
    self.warrior.instructions.clear()
    for i in range(clen):
      insn = self._get_inst(coresize = self.core_size)
      insn.opcode = OPCODES[self._OPCODE(_insn[i])]
      insn.amode = MODES[self._AMODE(_insn[i])]
      insn.bmode = MODES[self._BMODE(_insn[i])]
      insn.afield = int(_field[i,0])
      insn.bfield = int(_field[i,1])
      self.warrior.instructions.append(insn)

  def _get_obs_none(self):
    return (np.zeros(shape=self.dim_obs_insn, dtype=np.uint16), np.zeros(shape=self.dim_obs_field, dtype=np.uint16))

  def _get_reward_none(self):
    return 0

  def _get_obs_full(self):
    return (self.coredump[:,:,0], self.coredump[:,:,1:])

  def _get_obs_warrior(self):
    raise ValueError("not supported")

  def _get_reward(self):
    c = self.sum_proc.shape[0]
    for i in range(self.sum_proc.shape[1]):
      dat = self.sum_proc[:,i]
      print('w%d: end: %d max: %d avg: %f' % (i, dat[c-1], np.max(dat), np.average(dat)))
    w1 = np.array([(2**(10*(i+1)/c))/(2**10) for i in range(c)])
    w2 = np.array([-0.1]*self.num_players)
    w2[0] = 0.2
    r_proc = np.dot (np.dot (w1, self.sum_proc), w2)
    print('reward proc %f' % r_proc)
    r_dura = c * 0.05
    print('reward dura %f' % r_dura)
    reward = r_proc + r_dura
    print('reward %f' % reward)
    
    return reward

  def _reset(self):
    self.turn = 0
    self.cycles = 0
    self.sum_proc = np.ones((1, self.num_players), dtype=np.int32)
    self.coredump = np.zeros((self.dim_obs_sample, self.core_size, 3))

  def reset(self):
    self._reset()
    self.seed = np.random.randint(10000000)
    self.winners = []
    self.opponent = self.opponents[1]
    self.warrior = Corewar.Warrior()
    self.warrior.name = 'RL_Imp'
    self.warrior.author = 'my computer'
    self.warrior.start = int(self.max_length / 2)
    for i in range(self.max_length):
      insn = self._get_inst(self.core_size)
      insn.opcode = OPCODES[0]
      insn.amode = MODES[0]
      insn.bmode = MODES[0]
      insn.afield = 0
      insn.bfield = 0 
      self.warrior.instructions.append(insn)
    res = self.mars.open((self.warrior, self.opponent), seed = self.seed)
    self.coredump[0, :] = np.array(self.mars.dumpcore(), dtype=np.uint16)
    self.mars.stop()
    return self._get_obs()

  def render(self, mode='human', close=False):
    img = self._get_image()
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      return img
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None


if __name__=="__main__":
  env = CoreWarEnv(std='icws_88')
  print('init\n')
  obs = env.reset()
  print('reset\n')
  steps = 0
  total_reward = 0
  wins = 0
  loses = 0
  ties = 0
  epoch = int(1e7)
  for _ in range(epoch):
    print('step %d' % _)
    env.render()
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
  for i in len(env.winners):
    print('winner %d' % i)
    for j in len(env.winners[i]):
      print(env.winners[i][j])