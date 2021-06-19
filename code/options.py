import argparse
from multiprocessing import cpu_count
import os

board_size = 8
path = os.path.abspath('..')
path += "\\model"
file = path + "\\current_policy_" + str(board_size) + ".model"

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000, help="number of rounds of training")
parser.add_argument('--width', type=int, default=board_size, help="width of board")
parser.add_argument('--height', type=int, default=board_size, help="height of board")
parser.add_argument('--n_in_row', type=int, default=5, help="game rule: 5 in row to win")
parser.add_argument('--learn_rate', type=float, default=2e-3, help="Learning rate")
parser.add_argument('--buffer_size', type=int, default=1500, help="memory capacity")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--check_freq', type=int, default=30, help="frequency of saving model")
parser.add_argument('--method', type=str, default="asy", help="training algorithm")
parser.add_argument('--num_workers', type=int, default=cpu_count() - 1, help="worker number in asy env")
parser.add_argument('--model_file', type=str, default=None, help="initial model")
parser.add_argument('--save_file', type=str, default=file, help="initial model")
args = parser.parse_args()

if os.path.isfile(file):
    args.model_file = file



