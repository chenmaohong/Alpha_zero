import random
import numpy as np
from collections import deque
# from __future__ import print_function
import torch
from mcts_alphaZero import MCTSPlayer
from game import Board, Game
from policy_value_net_pytorch import PolicyValueNet
from multiprocessing import Pool, cpu_count

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainPipeline(object):
    def __init__(self, init_model=None):
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)

        self.temp = 1.0
        self.c_puct = 5
        self.n_playerout = 400
        self.learn_rate = 2e-3
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.check_freq = 50  # 保存模型的频率
        self.game_batch_num = 3000  # 训练更新的次数
        if init_model:  # 有无初始模型
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playerout, is_selfplay=1)

    #
    def run(self):
        try:
            for i in range(self.game_batch_num):
                episode_len = self.collect_selfplay_data()
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print(("batch i:{}, episode_len:{}, loss:{:.4f}, entropy:{:.4f}").format(i + 1, episode_len, loss,
                                                                                             entropy))
                else:
                    print("batch i:{}, episode_len:{}".format(i + 1, episode_len))
                if (i + 1) % self.check_freq == 0:
                    self.policy_value_net.save_model('./current_policy_15.model')
        except KeyboardInterrupt:
            print('\n quit')

    def asy_run(self):
        try:
            self.collect_selfplay_data()
            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_update()
                return loss, entropy
            else:
                return -1, -1
        except KeyboardInterrupt:
            print('\n quit')

    # def task(self):
    #     for i in range(self.game_batch_num // cpu_count()):
    #         episode_len = self.collect_selfplay_data()
    #         if len(self.data_buffer) > self.batch_size:
    #             loss, entropy = self.policy_update()
    #             print(("batch i:{}, episode_len:{}, loss:{:.4f}, entropy:{:.4f}").format(i + 1, episode_len, loss,
    #                                                                                      entropy))
    #         else:
    #             print("batch i:{}, episode_len:{}".format(i + 1, episode_len))
    #         if (i + 1) % self.check_freq == 0:
    #             self.policy_value_net.save_model('./current_policy.model')

    def collect_selfplay_data(self):
        """ collect self-play data for training """
        winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
        play_data = list(play_data)[:]
        episode_len = len(play_data)
        # augment the data
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)
        return episode_len

    def get_equi_data(self, play_data):
        """ play_data:[(state, mcts_prob, winner_z),..., ...]"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        loss = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate)
        return loss

        # loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate)
        # return loss, entropy


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
