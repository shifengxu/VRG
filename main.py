import os
from utils import log_info
from schedule.vrg_scheduler import VrgScheduler

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lp', type=float, default=0.01, help='learning portion')
    parser.add_argument("--reg_lambda", type=float, default=100, help='regularizer lambda')
    args = parser.parse_args()

    # add device
    args.device = 'cpu'
    return args

def main():
    args = parse_args()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"args: {args}")

    vs = VrgScheduler(args)
    old_trajectory_file = "./dpm_solver_original_trajectory/dpm_alphaBar_1-010-time_quadratic.txt"
    f_name = os.path.split(old_trajectory_file)[1]
    new_trajectory_file = f"./new_{f_name}"
    vs.load_trajectory_and_train(old_trajectory_file, new_trajectory_file)

    return 0

if __name__ == "__main__":
    main()
