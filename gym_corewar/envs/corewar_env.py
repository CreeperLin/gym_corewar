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
        seed=None,                                                # random seed
        std='icws_88',                                            # ICWS standard
        act_type='prog_discrete',                                 # action space type, can be of ['prog'|'direct]_['discrete'|'continuous'|'hybrid']
        obs_type='full',                                          # observation space type, can be of 'full', the dumps of the full core
        coresize=8000,                                            # size of the core
        fieldrange=None,                                          # range of the fields of instructions in action space
        maxprocesses=8000,                                        # maximum number of warriors in the core
        maxcycles=10000,                                          # maximum core cycles before tie
        maxsteps=1000,                                            # maximum steps before 'done' is true
        dumpintv=5,                                               # interval of the core dump in cycles
        dumprange=None,                                           # range of the core dump, away from address 0
        mindistance=100,                                          # minimum distance between warriors
        maxlength=100,                                            # maximum length of all warriors
        actmaxlength=25,                                          # maximum length of the warrior specified by action space
        pspacesize=None,                                          # pspace type
        opponents='warriors/88/simplified/Imp.red',               # list of file paths to opponent warriors
        initwarrior=None,                                         # initial warrior
        warriorname='RL_Imp',                                     # name of the warrior
        warriorauthor='my computer',                              # author of the warrior
        numplayers=2,                                             # number of corewar players
        verbose=False,                                            # output debug msg if true
        randomize=False,                                          # randomize initial positions of warriors before each run
        recordvideo=False,                                        # record videos based on result
        rewardfunc=None,                                          # custom reward function
        ):
    if (not opponents and not isinstance(opponents, str) and len(opponents) == 0):
      raise ValueError("specify path to opponent warriors")
    if (numplayers>2):
      raise ValueError("multi-warrior not supported")
    if (mindistance>coresize):
      raise ValueError("mindistance > coresize")
    self.viewer = None
    self.core_size = coresize
    if not fieldrange:
      self.field_range = coresize
    else:
      self.field_range = fieldrange % coresize
    self.max_proc = maxprocesses
    self.max_cycle = maxcycles
    self.max_steps = maxsteps
    self.max_length = maxlength
    self.max_length_ac = actmaxlength
    if self.max_length_ac > self.max_length:
      self.max_length_ac = self.max_length
    self.min_dist = mindistance
    if self.min_dist < self.max_length:
      self.min_dist = self.max_length
    self.obs_dump_intv = dumpintv
    if not dumprange:
      dumprange = self.core_size
    elif (dumprange > self.core_size // 2):
      dumprange = self.core_size // 2
    self.obs_dump_range = dumprange
    self.num_players = numplayers
    self.verbose = verbose
    self.randomize = randomize
    self.record_video = recordvideo
    self._reward_func = self._default_reward
    if (rewardfunc):
      self._reward_func = rewardfunc
    if (not seed):
      self.seed(0)
    else:
      self.seed(seed)

    if std=='icws_88':
      self.dim_opcode = dim_opcode_88
      self.dim_modifiers = dim_modifiers_88
      self.dim_addrmodes = dim_addrmodes_88
      self._get_inst = Instruction88
      self._OPCODE = OPCODE88
      self._AMODE = AMODE88
      self._BMODE = BMODE88
      # self._MODIFIER = MODIFIER
      self.parser = Corewar.Parser(coresize=self.core_size,
                              maxprocesses=self.max_proc,
                              maxcycles=self.max_cycle,
                              maxlength=self.max_length,
                              mindistance=self.min_dist,
                              standard=Corewar.STANDARD_88)
      self.mars = Corewar.Benchmarking.MARS_88(coresize = self.core_size,
                              maxprocesses = self.max_proc,
                              maxcycles = self.max_cycle,
                              mindistance = self.min_dist,
                              maxlength = self.max_length)
    elif std=='icws_94_nop':
      self.dim_opcode = dim_opcode_94_nop
      self.dim_modifiers = dim_modifiers_94_nop
      self.dim_addrmodes = dim_addrmodes_94_nop
      self._get_inst = Instruction
      self._OPCODE = OPCODE
      self._AMODE = AMODE
      self._BMODE = BMODE
      self._MODIFIER = MODIFIER
      self.parser = Corewar.Parser(coresize=self.core_size,
                              maxprocesses=self.max_proc,
                              maxcycles=self.max_cycle,
                              maxlength=self.max_length,
                              mindistance=self.min_dist,
                              standard=Corewar.STANDARD_94_NOP)
      self.mars = Corewar.Benchmarking.MARS_94nop(coresize = self.core_size,
                              maxprocesses = self.max_proc,
                              maxcycles = self.max_cycle,
                              mindistance = self.min_dist,
                              maxlength = self.max_length)
    elif std=='icws_94':
      raise ValueError("standard icws94 not supported")
    else:
      raise ValueError("invalid standard")

    self.opponents = []
    if isinstance(opponents, str):
      opponents = [opponents]
    for i in range(len(opponents)):
      self.log('reading warrior in %s' % (opponents[i]))
      self.opponents.append(self.parser.parse_file(opponents[i]))

    self.init_warrior = initwarrior
    self.wname = warriorname
    self.wauthor = warriorauthor
    self._reset_warrior()

    self.num_insn = self.dim_opcode*self.dim_addrmodes*self.dim_addrmodes
    if (act_type=='direct_discrete'):
      self.dim_action_insn = (self.max_length_ac, )
      self.dim_action_field = (self.max_length_ac, 2)
      self._parse_act = self._parse_act_direct_disc
      self.action_space = MultiDiscrete([self.num_insn]*self.max_length_ac + 
                                        [self.field_range]*self.max_length_ac + 
                                        [self.field_range]*self.max_length_ac)
    elif (act_type=='direct_continuous'):
      self.dim_action_insn = (self.max_length_ac, self.num_insn)
      self.dim_action_field = (self.max_length_ac, 2)
      self._parse_act = self._parse_act_direct_cont
      ilow = np.full(self.dim_action_insn, 0.0)
      ihigh = np.full(self.dim_action_insn, 1.0)
      flow = np.full(self.dim_action_field, 0.0)
      fhigh = np.full(self.dim_action_field, self.field_range-1)
      low = np.concatenate((ilow, flow), axis=1)
      high = np.concatenate((ihigh, fhigh), axis=1)
      self.action_space = Box(low=low, high=high, dtype=np.float32)
    elif (act_type=='direct_hybrid'):
      self.dim_action_field = (self.max_length_ac, 2)
      self._parse_act = self._parse_act_direct_hybd
      self.action_space = Tuple((
                                MultiDiscrete([self.num_insn]*self.max_length_ac),
                                Box(low=0, high=self.field_range-1,
                                    shape=self.dim_action_field, dtype=np.uint16)))
    elif (act_type=='prog_discrete'):
      self._parse_act = self._parse_act_prog_disc
      self.action_space = MultiDiscrete([dim_progress_act, self.max_length_ac,
                                        self.num_insn, self.field_range, self.field_range])
    elif (act_type=='prog_continuous'):
      self._parse_act = self._parse_act_prog_cont
      self.action_space = Box(low=np.array([0.0]*dim_progress_act + [0.0] + [0.0]*self.num_insn + [0.0]*2),
                              high=np.array([1.0]*dim_progress_act + [self.max_length_ac] + [1.0]*self.num_insn + [self.field_range]*2), dtype=np.float32)
    elif (act_type=='prog_hybrid'):
      self._parse_act = self._parse_act_prog_hybd
      self.action_space = Tuple((
                                Discrete(dim_progress_act),
                                Box(low=0, high=self.max_length_ac-1, shape=(1,), dtype=np.uint16),
                                Discrete(self.num_insn),
                                Box(low=0, high=self.field_range-1, shape=(2,), dtype=np.uint16)
                                ))
    else:
      raise ValueError("invalid action space type")
                            
    self.dim_obs_sample = int(self.max_cycle // self.obs_dump_intv)
    if (obs_type == 'full'):
      self._get_obs = self._get_obs_full
      self.dim_obs_insn  = (self.dim_obs_sample, self.obs_dump_range * 2, )
      self.dim_obs_field = (self.dim_obs_sample, self.obs_dump_range * 2, 2)
      self.observation_space = Dict({
                                'op': Box(low=0, high=self.dim_opcode-1, 
                                    shape=self.dim_obs_insn, dtype=np.uint8),
                                'amode': Box(low=0, high=self.dim_addrmodes-1, 
                                    shape=self.dim_obs_insn, dtype=np.uint8),
                                'bmode': Box(low=0, high=self.dim_addrmodes-1, 
                                    shape=self.dim_obs_insn, dtype=np.uint8),
                                'fields':Box(low=0, high=self.core_size-1,
                                    shape=self.dim_obs_field, dtype=np.uint16),
                                'proc': Box(low=0, high=self.core_size-1, 
                                            shape=(self.dim_obs_sample, self.max_proc * self.num_players),
                                            dtype=np.uint16)
                                })
    elif (obs_type == 'warriors'):
      raise ValueError("obs space type not supported")
    else:
      raise ValueError("invalid observation space type")

  def log(self,*args):
    if (self.verbose): print(*args)

  def _get_image(self, sc=10, idx=0):
    row = int(np.sqrt(self.core_size))
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
    cd = self._OPCODE(self.coredump[idx,:,0])
    for i in range(self.core_size):
      p = palette[cd[i]]
      ys = i // row
      xs = i % row
      img[ys*sc:(ys+1)*sc,xs*sc:(xs+1)*sc,:] = p
    for i in range(self.max_proc):
      t = self.procdump[idx,i]
      if (t==0): continue
      t-=1
      ys = t // row
      xs = t % row
      img[ys*sc : ys*sc+sc//2, xs*sc : xs*sc+sc//2, :] = [i % 256, (i**2) % 256, (i**3) % 256]
    for i in range(self.max_proc):
      t = self.procdump[idx,i+self.max_proc]
      if (t==0): continue
      t-=1
      ys = t // row
      xs = t % row
      img[ys*sc+sc//2:ys*sc+sc, xs*sc+sc//2:xs*sc+sc,:] = [i % 256, (i**2) % 256, (i**3) % 256]
    return img

  def step(self, action):
    self._reset()
    done = False

    self._parse_act(action)

    self.log('\n%s fighting with %s (seed=%d)' % (self.warrior.name, self.opponent.name, self._seed))

    res = self.mars.open((self.warrior, self.opponent), seed = self._seed)
    
    if (res!=0):
      raise ValueError("error opening MARS")

    self.match = -1
    while (True):
      if self.cycles >= self.max_cycle:
        break
      if (self.cycles % self.obs_dump_intv == 0):
        # self.log('dump %d' % obs_cnt)
        self.coredump[self.obs_cnt, :] = np.array(self.mars.dumpcore(), dtype=np.uint16)
        self.procdump[self.obs_cnt, :] = np.array(self.mars.dumpproc(), dtype=np.uint16) + 1
        self.obs_cnt+=1
      
      res = self.mars.step()
      if (res==0):
        self.log("warrior %d lose" % self.turn)
        self.match = self.turn
        break
      else:
        # self.log("warrior %d: %d" % (self.turn, res))
        if (self.cycles % self.num_players == 0):
          tmp = [0] * self.num_players
          tmp[0] = res
          self.sum_proc = np.append(self.sum_proc, [tmp], axis=0)
        else:
          self.sum_proc[int(self.cycles // self.num_players)+1][self.turn] = res


      self.cycles += 1
      self.turn = (self.turn + 1) % self.num_players
    
    self.mars.stop()
    self.log('cycle: %d' % (self.cycles))

    if (self.match>0):
      print('winner %d!' % self.wincount)
      print('cycle: %d seed %d' % (self.cycles, self._seed))
      c = self.sum_proc.shape[0]
      for i in range(self.sum_proc.shape[1]):
        dat = self.sum_proc[:,i]
        print('w%d: end: %d max: %d avg: %f' % (i, dat[c-1], np.max(dat), np.average(dat)))
      print(self.warrior)
      if (self.cycles > self.max_cycle // 2):
        self.winners.append(self.warrior.instructions.copy())
        if (self.record_video):
          self.record_coredump()
          with open('./winner'+str(wincount)+'.red','w') as f:
            f.write(str(self.warrior))
      self.wincount += 1

    self.steps += 1
    if (self.steps >= self.max_steps):
      done = True
    return self._get_obs(), self._get_reward(), done, {'match':self.match}

  def _get_softmax(self, z, a):
    return np.argmax(np.exp(z)/sum(np.exp(z)), axis=a)

  def _parse_act_prog_disc(self, action):
    _pact, _pnum, _insn, _afield, _bfield = action
    self._apply_act_prog(_pact, _pnum, _insn, _afield, _bfield)

  def _parse_act_prog_cont(self, action):
    _pact = self._get_softmax(action[:3], 0)
    _pnum = action[3]
    _insn = self._get_softmax(action[4:4+self.num_insn], 0)
    _afield = action[-2]
    _bfield = action[-1]
    self._apply_act_prog(_pact, _pnum, _insn, _afield, _bfield)

  def _parse_act_prog_hybd(self, action):
    _pact, _pnum, _insn, _field = action
    _afield = _field[0]
    _bfield = _field[1]
    self._apply_act_prog(_pact, _pnum, _insn, _afield, _bfield)

  def _apply_act_prog(self, act, idx, insn, afield, bfield):
    opcode = OPCODES[self._OPCODE(insn)]
    amode = MODES[self._AMODE(insn)]
    bmode = MODES[self._BMODE(insn)]
    idx = int(idx)
    if idx >= self.max_length_ac:
      idx = self.max_length_ac-1
    if act == 0: # NOOP
      return
    elif act == 1: # INSERT FRONT
      _insn = self._get_inst(coresize = self.core_size)
      _insn.opcode = opcode
      _insn.amode = amode
      _insn.bmode = bmode
      _insn.afield = int(afield)
      _insn.bfield = int(bfield)
      self.insns[idx] = _insn
    elif act == 2: # DELETE FRONT
      self.insns[idx] = None
    else:
      raise ValueError("undefined prog_act")

    self.warrior.instructions.clear()
    for i in range(len(self.insns)):
      if (not self.insns[i]):
        # self.warrior.instructions.append(self._get_inst(self.core_size))
        continue
      self.warrior.instructions.append(self.insns[i])
    if (len(self.warrior.instructions)==0):
      self.warrior.instructions.append(self._get_inst(self.core_size))

  def _parse_act_direct_disc(self, action):
    _insn = action[0:self.max_length_ac]
    _afield = action[self.max_length_ac:2*self.max_length_ac]
    _bfield = action[2*self.max_length_ac:3*self.max_length_ac]
    self._apply_act_direct(_insn, _afield, _bfield)

  def _parse_act_direct_cont(self, action):
    _insn = self._get_softmax(action[:,:-2], 1)
    _afield = action[:,-2]
    _bfield = action[:,-1]
    self._apply_act_direct(_insn, _afield, _bfield)

  def _parse_act_direct_hybd(self, action):
    _insn, _field = action
    _afield = _field[:,0]
    _bfield = _field[:,1]
    self._apply_act_direct(_insn, _afield, _bfield)

  def _apply_act_direct(self, insn, afield, bfield):
    self.warrior.instructions.clear()
    op = self._OPCODE(insn)
    amode = self._AMODE(insn)
    bmode = self._BMODE(insn)
    for i in range(self.max_length_ac):
      _insn = self._get_inst(coresize = self.core_size)
      _insn.opcode = OPCODES[op[i]]
      _insn.amode = MODES[amode[i]]
      _insn.bmode = MODES[bmode[i]]
      _insn.afield = int(afield[i])
      _insn.bfield = int(bfield[i])
      self.warrior.instructions.append(_insn)
    self.warrior.start = int(self.max_length_ac / 2)

  def _get_obs_none(self):
    return (np.zeros(shape=self.dim_obs_insn, dtype=np.uint16), np.zeros(shape=self.dim_obs_field, dtype=np.uint16))

  def _get_reward_none(self):
    return 0

  def _get_obs_full(self):
    cdwindow = np.concatenate((self.coredump[:,-self.obs_dump_range:,:], self.coredump[:,:self.obs_dump_range,:]), axis=1)
    insns = cdwindow[:,:,0]
    # return {'insns':insns, 'fields':self.coredump[:,:,1:]}
    return {'op':self._OPCODE(insns), 'amode':self._AMODE(insns), 'bmode':self._BMODE(insns), 'fields':cdwindow[:,:,1:],
            'proc':self.procdump}

  def _get_obs_warrior(self):
    raise ValueError("not supported")

  def _default_reward(self, match, cycles, sum_proc):
    c = sum_proc.shape[0]
    for i in range(sum_proc.shape[1]):
      dat = sum_proc[:,i]
      self.log('w%d: end: %d max: %d avg: %f' % (i, dat[c-1], np.max(dat), np.average(dat)))
    w1 = np.array([(1.2**(2*(i+1)/c))/(1.2**2) for i in range(c)])
    w2 = np.array([-1.0]*self.num_players)
    w2[0] = 1.0
    r_proc = 100 * np.dot (np.dot (w1, sum_proc), w2) / (cycles + 1) / 2
    self.log('score proc %f' % r_proc)
    r_dura = 100 * cycles / self.max_cycle
    self.log('score dura %f' % r_dura)
    r_goal = 1 if match > 0 else 0 if match == 0 else 0.25
    r_goal = 1000 * r_goal * cycles / self.max_cycle
    self.log('score goal %f' % r_goal)
    score = r_proc + r_dura + r_goal
    ret = score - self.last_score + 1000 * (1 if match > 0 else 0) * cycles / self.max_cycle
    self.log('score %f reward %f' % (score, ret))
    self.last_score = score
    return ret

  def _get_reward(self):
    return self._reward_func(match=self.match, cycles=self.cycles, sum_proc=self.sum_proc)

  def _reset(self):
    self.turn = 0
    self.cycles = 0
    self.obs_cnt = 0
    self.sum_proc = np.ones((1, self.num_players), dtype=np.uint16)
    self.coredump = np.zeros((self.dim_obs_sample, self.core_size, 3), dtype=np.uint16)
    self.procdump = np.zeros((self.dim_obs_sample, self.max_proc * self.num_players), dtype=np.uint16)
    if (self.randomize):
      self._seed = np.random.randint(low=self.min_dist, high=self.core_size)

  def reset(self):
    self._reset()
    self.steps = 0
    self.winners = []
    self.wincount = 0
    self.last_score = 0
    self._reset_warrior()
    self.opponent = self.opponents[0]
    res = self.mars.open((self.warrior, self.opponent), seed = self._seed)
    self.coredump[0, :] = np.array(self.mars.dumpcore(), dtype=np.uint16)
    self.mars.stop()
    return self._get_obs()
  
  def _reset_warrior(self):
    if (not self.init_warrior):
      self.warrior = Corewar.Warrior()
      self.warrior.start = 0
      insn = self._get_inst(self.core_size)
      self.warrior.instructions.append(insn)
    else:
      self.warrior = self.parser.parse_file(self.init_warrior)
    self.warrior.name = self.wname
    self.warrior.author = self.wauthor
    self.insns = []
    for i in range(len(self.warrior.instructions)):
      self.insns.append(self.warrior.instructions[i])
    for i in range(self.max_length-len(self.insns)):
      self.insns.append(None)

  def render(self, mode='human'):
    if mode == 'ansi':
      from six import StringIO
      outfile = StringIO()
      outfile.write(str(self.warrior))
      outfile.write('\n')
      return outfile
    if mode == 'rgb_array':
      img = self._get_image(idx=self.obs_cnt-1)
      return img
    if mode == 'human':
      img = self._get_image(idx=self.obs_cnt-1)
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)
      return self.viewer.isopen

  def record_coredump(self):
    import cv2
    fps = 10
    name = 'winner'+str(self.wincount)
    sc = 20
    row = int(np.sqrt(self.core_size))
    img_w = row * sc
    img_h = self.core_size // row * sc
    size = (img_w, img_h)
    out = cv2.VideoWriter(str(name) + '.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    for i in range(self.obs_cnt):
      frame = self._get_image(sc=sc,idx=i)
      out.write(frame)
    out.release()

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None

  def seed(self, s):
    np.random.seed(s)
    self._seed = np.random.randint(low=self.min_dist, high=self.core_size)
