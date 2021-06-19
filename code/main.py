from train_method import asy_train, cen_mcts
from options import args
import time

if __name__ == '__main__':
    start = time.time()
    if args.method == "cen":
        train_pipe = cen_mcts()
    else:
        if args.method == "asy":
            train_pipe = asy_train(args.num_workers, args.width, args.height)
    train_pipe.run()
    end = time.time()
    print("The training time is {}".format(end - start))
