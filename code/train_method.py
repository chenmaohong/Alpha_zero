from train import TrainPipeline
from policy_value_net_pytorch import PolicyValueNet, Net
import copy
from multiprocessing import Pool, cpu_count, set_start_method
import torch.optim as optim
import torch
from options import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class worker(TrainPipeline):
    def __init__(self, global_net):
        super(worker, self).__init__(global_net)
        self.policy_value_net.policy_value_net = copy.deepcopy(global_net)


class asy_train(object):
    def __init__(self, num_agent, board_width, board_height):
        self.num = num_agent
        self.round = 1000
        self.l2_const = 1e-4
        self.global_net = Net(board_width, board_height).to(device)
        self.optimizer = optim.Adam(self.global_net.parameters(), weight_decay=self.l2_const)
        if args.model_file:
            net_params = torch.load(args.model_file)
            self.global_net.load_state_dict(net_params)
        self.agent_set = list()
        for i in range(num_agent):
            self.agent_set.append(worker(self.global_net))

    def run(self):
        set_start_method('spawn')
        p = Pool(processes=self.num)
        for i in range(self.num):
            p.apply_async(self.task, args=(i,))
        p.close()
        p.join()

    def task(self, i):
        round = args.epochs // self.num
        for k in range(0, round):
            self.agent_set[i].policy_value_net.policy_value_net = copy.deepcopy(self.global_net)
            loss = self.agent_set[i].asy_run()
            print("Index:{}, batch:{}, loss:{:.4f}".format(i, k + 1, loss.data))
            if (k * self.num + i + 1) % args.check_freq == 0:
                torch.save(self.global_net.state_dict(), args.save_file, _use_new_zipfile_serialization=False)


class imp_train(object):
    pass


class cen_mcts(object):
    def __init__(self):
        self.runner = TrainPipeline()

    def run(self):
        self.runner.run()
