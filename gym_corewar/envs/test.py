import sys
import gc
import os
import numpy as np
import random
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

def fill(warrior):
  warrior.instructions.clear()
  for i in range(maxlength):
    insn = Instruction88(coresize = coresize)
    ins = random.randint(0, num_insn-1)
    insn.opcode = OPCODES[OPCODE88(ins)]
    insn.amode = MODES[AMODE88(ins)]
    insn.bmode = MODES[BMODE88(ins)]
    # insn.afield = 0
    # insn.bfield = 0
    insn.afield = int(random.randint(0, coresize-1))
    insn.bfield = int(random.randint(0, coresize-1))
    warrior.instructions.append(insn)

if __name__=="__main__":
  coresize=8000
  maxprocesses=8000
  maxcycles=10
  dumpintv=100
  mindistance=100
  maxlength=100
  num_players = 2
  parser = Corewar.Parser(coresize=coresize,
                              maxprocesses=maxprocesses,
                              maxcycles=maxcycles,
                              maxlength=maxlength,
                              mindistance=mindistance,
                              standard=Corewar.STANDARD_88)
  mars = Corewar.Benchmarking.MARS_88(coresize = coresize,
                              maxprocesses = maxprocesses,
                              maxcycles = maxcycles,
                              mindistance = mindistance,
                              maxlength = maxlength)
  dim_obs_sample = int(maxcycles // dumpintv)
  num_insn = 11*4*4
  opponents = []
  oppo = ('warriors/88/Imp.red',
    'warriors/88/Dwarf.red',
    'warriors/88/The_Seed.red',
    'warriors/88/MaxProcess.red'
    )
  for i in range(len(oppo)):
    print('reading warrior in %s' % (oppo[i]))
    opponents.append(parser.parse_file(oppo[i]))
  warrior = Corewar.Warrior()
  warrior.name = 'RL_Imp'
  warrior.author = 'my computer'
  warrior.start = int(maxlength / 2)
  turn = 0
  cycles = 0
  sum_proc = np.ones((1, num_players), dtype=np.int32)
  coredump = np.zeros((dim_obs_sample, coresize, 3))
  # epoch = int(10)
  epoch = int(1e7)
  fill(warrior)
  for _ in range(epoch):
    fill(warrior)
    print('epoch %d' % _)
    # mars.run((opponents[0], opponents[1]), seed = 3423423)
    # continue
    # res = mars.open((opponents[0], opponents[1]), seed = 3423423)
    continue
    res = mars.open((warrior, opponents[1]), seed = 3423423)
    res = mars.step()
    res = mars.stop()
    continue
    obs_cnt = 0
    while (True):
      res = mars.step()
      
      if (res==0):
        print("warrior %d lose" % turn)
        match = turn
        break
      if cycles >= maxcycles:
        break
      else:
        # print("warrior %d: %d" % (turn, res))
        if (cycles % num_players == 0):
          tmp = [0] * num_players
          tmp[0] = res
          sum_proc = np.append(sum_proc, [tmp], axis=0)
        else:
          sum_proc[int(cycles // num_players)+1][turn] = res

      cycles += 1
      turn = (turn + 1) % num_players

      if (cycles % dumpintv == 0):
        # print('dump %d' % (cycles))
        # coredump[obs_cnt, :] = np.array(mars.dumpcore(), dtype=np.uint16)
        obs_cnt+=1
    
    mars.stop()
    print('cycle: %d' % (cycles))
    turn = 0
    cycles = 0
    sum_proc = np.ones((1, num_players), dtype=np.int32)
    coredump = np.zeros((dim_obs_sample, coresize, 3))
