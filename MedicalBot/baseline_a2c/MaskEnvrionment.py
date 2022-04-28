import gym
from gym import spaces
import numpy as np
import random
import logging
from tkinter import *
from tkinter import messagebox
import json
import random
from NLP import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from NetModel import NeuralNet
import MyDialogue
logging.basicConfig(level=logging.INFO, filename='save.txt', filemode='w')


class MedicalEnvrionment(gym.Env):

    # 把贝叶斯网络直接当成actor，  适合NN
    def __init__(self, slot_set, start_set, max_turn=22, flag='train', disease_num=4):
        self.max_turn = max_turn
        # self.slot_set = slot_set
        self.slot_dict = {v: k for k, v in enumerate(slot_set)}
        self.num2slot = {v: k for k, v in self.slot_dict.items()}
        self.start_set = start_set
        self.action_space = spaces.Discrete(len(self.slot_dict))
        # 0表示未询问， -1 表示没有  1 表示有, 2表示已经问过了
        self.observation_space = spaces.Box(low=-1, high=2, shape=(len(self.slot_dict), ))
        self.state = None
        self.turn = 0
        self.goal = None
        self.action_mask = np.zeros(len(self.slot_dict))
        self.flag = flag
        self.disease_num = disease_num
        self.goal_num = 0
        self.mate_num = 0
        self.action_name = ''
        self.mess = ''
        self.active = False

        if self.flag == 'train':
            self.goal_num = 0

    def trans_self_report(self, report_dict):
        true_list, false_list = [], []
        for k, v in report_dict.items():
            if v:
                true_list.append(self.slot_dict[k])
            else:
                false_list.append(self.slot_dict[k])
        return true_list, false_list

    def seed(self, seed=None):
        if self.flag == "test":
            print("sedd： ",seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.manual_seed(seed)
        return None

    def reset(self):   # 第一次预测的时候是没有mask的，这导致训练和测试有一点点相差
        self.turn = 0
        self.mate_num = 0
        if self.flag == 'train':
            self.goal = random.choice(self.start_set)
            self.goal_num = 0
        else:
            # print(self.goal_num)
            if self.goal_num >= len(self.start_set):
                self.goal_num = 0
            self.goal = self.start_set[self.goal_num]
            self.goal_num += 1
        self.state = np.zeros(len(self.slot_dict))
        positive_list, negative_list = self.trans_self_report(self.goal['explicit_inform_slots'])
        im_positive_list, im_negative_list = self.trans_self_report(self.goal['implicit_inform_slots'])

        self.state[positive_list] = [1] * len(positive_list)
        self.state[negative_list] = [-1] * len(negative_list)
        if self.flag == 'train':
            self.action_mask = np.ones(len(self.slot_dict))   # 初始化是1, 表示全都给mask掉
            self.action_mask[im_positive_list+im_negative_list] = [0] * len(im_positive_list+im_negative_list)  # 把需要预测的留下
            self.action_mask[:self.disease_num] = [0] * self.disease_num
        else:
            self.action_mask = np.zeros(len(self.slot_dict))
            self.action_mask[positive_list+negative_list] = [1] * len(positive_list+negative_list)
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #print(self.action_mask)
        #print(action)
        #if self.flag == 'test':
        if int(action) < self.disease_num:
            mess = 'disease'
        else:
            mess = 'symptom'
        logging.info(str(self.goal_num)+"   "+mess+"@@@"+self.num2slot[action])
        #print(str(self.goal_num)+"   "+mess+"@@@"+self.num2slot[action])
        self.mess = mess
        action_name = self.num2slot[action]
        self.action_name = action_name

        self.action = action_name
        self.mess = mess
        samples = self.goal
        if self.action not in samples['explicit_inform_slots']:
            response = self.render(self)
        else:
            response = True

        if response:
            self.state[action] = 1
            if int(action) >= self.disease_num and self.action_mask[int(action)] == 0:
                self.mate_num += 1
        elif not response:
            self.state[action] = -1
            if int(action) >= self.disease_num and self.action_mask[int(action)] == 0:
                self.mate_num += 1
        else:
            self.state[action] = 0
        print(f"Stato: {self.state}")

        self.action_mask[int(action)] = 1
        self.turn += 1
        # done = (np.abs(x) + np.abs(y) <= 1) or (np.abs(x) + np.abs(y) >= 2 * self.L + 1)
        done = (self.turn > self.max_turn) or (int(action) < self.disease_num)
        done = bool(done)
        is_right = False
        if not done:
            reward = -1
        else:
            if action_name == self.goal['disease_tag']:
                reward = 44
                is_right = True
            else:
                reward = -22
        return self.state, reward, done, {"history": self.action_mask, "right": is_right, "turn": self.turn, 'ans': action, 'mate_num': self.mate_num,'done': done, 'disease': self.action_name}

    def close(self):
        return None

    def render(self, mode='human'):
        bot_name = "Sam"

        root = Tk()
        root.withdraw()

        if self.mess == "disease":
            return True

        answer = messagebox.askyesno(f'{bot_name} ask :', f" Do you have this {self.mess}? {self.action_name} :")

        return answer
