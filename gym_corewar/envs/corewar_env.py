import sys
import os
import numpy as np
import gym
from gym import error, spaces, utils
from gym.spaces import Discrete, Tuple, Box, MultiDiscrete, Dict
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
dim_progress_act = 3

class CoreWarEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

  def __init__(self,
        seed=None,
        std='icws_88', 
        act_type='progressive',
        obs_type='full',
        coresize=8000,
        maxprocesses=8000,
        maxcycles=10000,
        maxrounds=1000,
        dumpintv=5,
        mindistance=25,
        maxlength=25,
        pspacesize=None,
        opponents='warriors/88/simplified/Imp.red',
        initwarrior=None,
        warriorname='RL_Imp',
        warriorauthor='my computer',
        numplayers=2,
        ):
    if (not opponents and not isinstance(opponents, str) and len(opponents) == 0):
      raise ValueError("specify path to opponent warriors")
    if (numplayers>2):
      raise ValueError("multi-warrior not supported")
    self.viewer = None
    self.core_size = coresize
    self.max_proc = maxprocesses
    self.max_cycle = maxcycles
    self.max_rounds = maxrounds
    self.min_dist = mindistance
    self.max_length = maxlength
    self.obs_dump_intv = dumpintv
    self.num_players = numplayers
    self.seed(seed)
    if (not self._seed):
      self.seed(0)

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
    if isinstance(opponents, str):
      opponents = [opponents]
    for i in range(len(opponents)):
      print('reading warrior in %s' % (opponents[i]))
      self.opponents.append(self.parser.parse_file(opponents[i]))

    if (not initwarrior):
      self.warrior = Corewar.Warrior()
      self.warrior.start = 0
      insn = self._get_inst(self.core_size)
      insn.opcode = OPCODES[0]
      insn.amode = MODES[0]
      insn.bmode = MODES[0]
      insn.afield = 0
      insn.bfield = 0 
      self.warrior.instructions.append(insn)
    else:
      self.warrior = self.parser.parse_file(initwarrior)
    self.warrior.name = warriorname
    self.warrior.author = warriorauthor

    self.num_insn = self.dim_opcode*self.dim_addrmodes*self.dim_addrmodes
    if (act_type=='direct'):
      self.dim_action_insn = (self.max_length, )
      self.dim_action_field = (self.max_length, 2)
      self._parse_act = self._parse_act_direct
      # self.action_space = Tuple((
      #                           Box(low=0, high=self.num_insn-1,
      #                               shape=self.dim_action_insn, dtype=np.uint16),
      #                           # Box(low=-self.core_size/2+1, high=self.core_size/2,
      #                           Box(low=0, high=self.core_size-1,
      #                               shape=self.dim_action_field, dtype=np.uint16)))
      self.action_space = MultiDiscrete([self.num_insn]*self.max_length+[self.core_size]*self.max_length+[self.core_size]*self.max_length)
    elif (act_type=='progressive'):
      self.insns = []
      for i in range(len(self.warrior.instructions)):
        self.insns.append(self.warrior.instructions[i])
      for i in range(self.max_length-len(self.insns)):
        self.insns.append(None)
      self._parse_act = self._parse_act_prog
      self.action_space = MultiDiscrete([dim_progress_act, self.max_length,
                                        self.num_insn, self.core_size, self.core_size])
    else:
      raise ValueError("invalid action space type")
                            
    self.dim_obs_sample = int(self.max_cycle // self.obs_dump_intv)
    if (obs_type == 'full'):
      self._get_obs = self._get_obs_full
      self.dim_obs_insn  = (self.dim_obs_sample, self.core_size, )
      self.dim_obs_field = (self.dim_obs_sample, self.core_size, 2)
      self.observation_space = Dict({
                                'insns': Box(low=0, high=self.num_insn-1, 
                                    shape=self.dim_obs_insn, dtype=np.uint16),
                                # Box(low=-self.core_size/2+1, high=self.core_size/2,
                                'fields':Box(low=0, high=self.core_size-1,
                                    shape=self.dim_obs_field, dtype=np.uint16)
                                })
    elif (obs_type == 'warriors'):
      raise ValueError("obs space type not supported")
    else:
      raise ValueError("invalid observation space type")

  def _get_image(self):
    sc = 2
    row = 100
    img_w = row * sc
    img_h = self.core_size // row * sc
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    palette = np.array([
      [0,0,0],
      [255,255,255],
      [0,255,0],
      [0,0,255],
      [255,255,0],
      [255,0,255],
      [0,255,255],
      [255,0,0],
      [128,128,128],
      [128,255,255],
      [255,128,255],
      [255,255,128],
    ], dtype=np.uint8)
    # last_obs = self.cycles // self.obs_dump_intv
    cd = self.coredump[self.obs_cnt-1,:,0]
    for i in range(self.core_size):
      p = palette[self._OPCODE(int(cd[i]))]
      ys = i // row
      xs = i % row
      for y in range(ys*sc, (ys+1)*sc):
        for x in range(xs*sc, (xs+1)*sc):
          img[y][x] = p
    return img

  def step(self, action):
    self._reset()
    done = False

    self._parse_act(action)

    # res = self.mars.open((self.opponent, self.opponent), seed = self._seed)
    res = self.mars.open((self.warrior, self.opponent), seed = self._seed)
    
    if (res!=0):
      raise ValueError("error opening MARS")

    print('\n%s fighting with %s (%d)' % (self.warrior.name, self.opponent.name, self._seed))

    match = -1
    while (True):
      if self.cycles >= self.max_cycle:
        break
      if (self.cycles % self.obs_dump_intv == 0):
        # print('dump %d' % obs_cnt)
        self.coredump[self.obs_cnt, :] = np.array(self.mars.dumpcore(), dtype=np.uint16)
        self.obs_cnt+=1
      
      res = self.mars.step()
      if (res==0):
        print("warrior %d lose" % self.turn)
        match = self.turn
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
    
    self.mars.stop()
    print('cycle: %d' % (self.cycles))

    if (match>0):
      print('winner!')
      print(self.warrior)
      self.winners.append(self.warrior.instructions.copy())

    self.rounds += 1
    if (self.rounds >= self.max_rounds):
      done = True
    return self._get_obs(), self._get_reward(), done, {'match':match}

  def _parse_act_prog(self, action):
    _pact, _pnum, _insn, _afield, _bfield = action
    clen = len(self.warrior.instructions)
    opcode = OPCODES[self._OPCODE(_insn)]
    amode = MODES[self._AMODE(_insn)]
    bmode = MODES[self._BMODE(_insn)]
    afield = int(_afield)
    bfield = int(_bfield)
    idx = _pnum
    # if (idx >= clen):
      # idx = clen - 1
    if _pact == 0: # NOOP
      return
    elif _pact == 1: # INSERT FRONT
      # if (clen>=self.max_length):
        # return
      insn = self._get_inst(coresize = self.core_size)
      insn.opcode = opcode
      insn.amode = amode
      insn.bmode = bmode
      insn.afield = afield
      insn.bfield = bfield
      self.insns[idx] = insn
      # self.warrior.instructions.insert(idx, insn)
    elif _pact == 2: # DELETE FRONT
      # if clen > 1:
        # if (idx == clen):
          # idx -= 1
        # self.warrior.instructions.pop(idx)
      self.insns[idx] = None
    # elif _pact == 3: # MODIFY FRONT
      # self.warrior.instructions[i] = insn
    else:
      raise ValueError("undefined prog_act")

    self.warrior.instructions.clear()
    for i in range(len(self.insns)):
      if (not self.insns[i]): continue
      self.warrior.instructions.append(self.insns[i])
    if (len(self.warrior.instructions)==0):
      insn = self._get_inst(self.core_size)
      self.warrior.instructions.append(insn)

  def _parse_act_direct(self, action):
    # _insn, _field = action
    # _afield = _field[:,0]
    # _bfield = _field[:,1]
    _insn = action[0:self.max_length]
    _afield = action[self.max_length:2*self.max_length]
    _bfield = action[2*self.max_length:3*self.max_length]
    clen = _insn.shape[0]
    self.warrior.instructions.clear()
    for i in range(clen):
      insn = self._get_inst(coresize = self.core_size)
      insn.opcode = OPCODES[self._OPCODE(_insn[i])]
      insn.amode = MODES[self._AMODE(_insn[i])]
      insn.bmode = MODES[self._BMODE(_insn[i])]
      insn.afield = int(_afield[i])
      insn.bfield = int(_bfield[i])
      self.warrior.instructions.append(insn)
    self.warrior.start = int(self.max_length / 2)

  def _get_obs_none(self):
    return (np.zeros(shape=self.dim_obs_insn, dtype=np.uint16), np.zeros(shape=self.dim_obs_field, dtype=np.uint16))

  def _get_reward_none(self):
    return 0

  def _get_obs_full(self):
    return {'insns':self.coredump[:,:,0], 'fields':self.coredump[:,:,1:]}

  def _get_obs_warrior(self):
    raise ValueError("not supported")

  def _get_reward(self):
    c = self.sum_proc.shape[0]
    for i in range(self.sum_proc.shape[1]):
      dat = self.sum_proc[:,i]
      print('w%d: end: %d max: %d avg: %f' % (i, dat[c-1], np.max(dat), np.average(dat)))
    w1 = np.array([(1.2**(2*(i+1)/c))/(1.2**2) for i in range(c)])
    w2 = np.array([-1.0]*self.num_players)
    w2[0] = 2.0
    r_proc = np.dot (np.dot (w1, self.sum_proc), w2)
    print('reward proc %f' % r_proc)
    r_dura = self.cycles * 0.5
    print('reward dura %f' % r_dura)
    reward = r_proc + r_dura
    print('reward %f' % reward)
    
    return reward

  def _reset(self):
    self.turn = 0
    self.cycles = 0
    self.obs_cnt = 0
    self.sum_proc = np.ones((1, self.num_players), dtype=np.int32)
    self.coredump = np.zeros((self.dim_obs_sample, self.core_size, 3))

  def reset(self):
    self._reset()
    self.rounds = 0
    self.winners = []
    self.opponent = self.opponents[0]
    res = self.mars.open((self.warrior, self.opponent), seed = self._seed)
    self.coredump[0, :] = np.array(self.mars.dumpcore(), dtype=np.uint16)
    self.mars.stop()
    return self._get_obs()

  def render(self, mode='human'):
    from six import StringIO
    outfile = StringIO() if mode == 'ansi' else sys.stdout
    outfile.write(str(self.warrior))
    outfile.write('\n')
    if mode == 'rgb_array':
      img = self._get_image()
      return img
    elif mode == 'human':
      img = self._get_image()
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None

  def seed(self, s):
    np.random.seed(s)
    self._seed = np.random.randint(low=2*self.min_dist, high=self.core_size)
