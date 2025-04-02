import os
import torch
from torch import optim, Tensor

from utils import log_info
from .linear_simulator import LinearSimulator
from .vrg_alpha_model import VrgAlphaModel

def load_ts_alphabar_mse_from_file(f_path):
    if not os.path.exists(f_path):
        raise Exception(f"File not found: {f_path}")
    if not os.path.isfile(f_path):
        raise Exception(f"Not file: {f_path}")
    log_info(f"load_ts_alphabar_mse(): {f_path}...")
    with open(f_path, 'r') as f:
        lines = f.readlines()
    ts_arr, alphabar_arr, mse_arr = [], [], []
    data_lines = []
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        data_lines.append(line)
        arr = line.split('\t')
        ts, alphabar, mse = int(arr[0]), float(arr[1]), float(arr[2])
        ts_arr.append(ts)
        alphabar_arr.append(alphabar)
        mse_arr.append(mse)
    data_count = len(data_lines)
    log_info(f"  data_count  : {data_count}")
    for i in range(5):
        log_info(f"  data_lines[{i}]: {data_lines[i]}")
    for i in range(data_count - 5, data_count):
        log_info(f"  data_lines[{i}]: {data_lines[i]}")
    alphabar_arr = torch.tensor(alphabar_arr, dtype=torch.float64)
    mse_arr = torch.tensor(mse_arr, dtype=torch.float64)
    log_info(f"load_ts_alphabar_mse(): {f_path}...Done")
    return ts_arr, alphabar_arr, mse_arr

def load_trajectory_from_file(f_path):
    log_info(f"  load_trajectory_from_file()...")
    log_info(f"    {f_path}")
    with open(f_path, 'r') as f:
        lines = f.readlines()
    cnt_empty = 0
    float_arr, str_arr, comment_arr = [], [], []
    for line in lines:
        line = line.strip()
        if line == '':
            cnt_empty += 1
            continue
        if line.startswith('#'):
            comment_arr.append(line)
            continue
        # handle the data. Sample data:
        #    0.99970543
        #    0.99970543  : 0000.00561   <<< 2nd column is timestep in new version
        flt = float(line.split(':')[0].strip()) if ':' in line else float(line)
        float_arr.append(flt)
        str_arr.append(line)
    # for
    log_info(f"    cnt_empty  : {cnt_empty}")
    log_info(f"    cnt_comment: {len(comment_arr)}")
    log_info(f"    cnt_valid  : {len(float_arr)}")
    log_info(f"  load_trajectory_from_file()...Done")
    return float_arr, str_arr, comment_arr

def accumulate_variance(alpha: Tensor, alphabar: Tensor, weight_arr: Tensor, details=False):
    """
    accumulate variance from x_1000 to x_1.
    """
    # delta is to avoid torch error:
    #   RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
    # Or:
    #   the 2nd epoch will have output: tensor([nan, nan, ,,,])
    # Of that error, a possible reason is: torch tensor 0.sqrt()
    # So here, we make sure alpha > alphabar.
    delta = torch.zeros_like(alphabar)
    delta[0] = 1e-16
    coef = ((1-alphabar).sqrt() - (alpha+delta-alphabar).sqrt())**2
    numerator = coef * weight_arr
    sub_var = numerator / alphabar
    # sub_var *= alpha
    final_var = torch.sum(sub_var)
    if details:
        return final_var, coef, numerator, sub_var
    return final_var

class VrgScheduler:
    def __init__(self, args):
        log_info(f"VrgScheduler::__init__()...")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # we may use other MSE files.
        mse_file = os.path.join(cur_dir, 'MSE_vs_alpha_bar_cifar10.txt')
        ts_arr, alphabar_arr, mse_arr = load_ts_alphabar_mse_from_file(mse_file)
        self.vs = LinearSimulator(alphabar_arr, mse_arr)
        self.vs.to(args.device)
        self.lr           = args.lr   # learning-rate
        self.lp           = args.lp   # learning_portion
        self.n_epochs     = args.n_epochs
        self.reg_lambda   = args.reg_lambda   # regularizer lambda
        self.device       = args.device
        self.log_interval = 100
        log_info(f"  device       : {self.device}")
        log_info(f"  lr           : {self.lr}")
        log_info(f"  lp           : {self.lp}")
        log_info(f"  n_epochs     : {self.n_epochs}")
        log_info(f"  reg_lambda   : {self.reg_lambda}")
        log_info(f"  log_interval : {self.log_interval}")
        log_info(f"VrgScheduler::__init__()...Done")

    def load_trajectory_and_train(self, old_trajectory_file, new_trajectory_file):
        log_info(f"VrgScheduler::load_trajectory_and_train()...")
        log_info(f"  old_trajectory_file: {old_trajectory_file}")
        log_info(f"  new_trajectory_file: {new_trajectory_file}")
        alpha_bar, line_arr, c_arr = load_trajectory_from_file(old_trajectory_file)
        alpha_bar = alpha_bar[1:]  # ignore the first one, as it is for timestep 0
        line_arr  = line_arr[1:]
        c_arr = [c[1:] for c in c_arr]  # remove prefix '#'
        c_arr.insert(0, f" Old comments in file {old_trajectory_file}")
        _, idx_arr = self.vs(torch.tensor(alpha_bar, device=self.device), include_index=True)
        s_arr = [f"{line_arr[i]} : {idx_arr[i]:4d}" for i in range(len(alpha_bar))]
        s_arr.insert(0, "Old alpha_bar and its timestep, and estimated timestep in vs")
        c_arr = c_arr + [''] + s_arr

        new_msg_arr = [f"lr           : {self.lr}",
                       f"lp           : {self.lp}",
                       f"ori_ab_lowest: {alpha_bar[-1]}",
                       f"reg_lambda   : {self.reg_lambda}",
                       f"n_epochs     : {self.n_epochs}",
                       f"torch.seed() : {torch.seed()}"]  # message array
        c_arr = c_arr + [''] + new_msg_arr

        res = self.train(alpha_bar, c_arr, new_trajectory_file)
        log_info(f"VrgScheduler::load_trajectory_and_train()...Done")
        return res

    def train(self, alpha_bar_arr, msg_arr, output_file):
        log_info(f"VrgScheduler::train()...")
        # cpe: cumulative-prediction-error
        # reg: regularizer
        # calculate cpe for original trajectory
        aacum = torch.tensor(alpha_bar_arr, device=self.device)
        a_tmp = aacum[1:] / aacum[:-1]
        alpha = torch.cat([aacum[0:1], a_tmp], dim=0)
        weight_arr, idx_arr = self.vs(aacum, include_index=True)
        cpe_ori, _, _, _ = accumulate_variance(alpha, aacum, weight_arr, True)
        ori_lowest_alphabar = alpha_bar_arr[-1]
        # ori_lowest_alphabar = 0.0001

        model = VrgAlphaModel(alpha_bar_list=alpha_bar_arr, learning_portion=self.lp)
        log_info(f"  VrgAlphaModel.to({self.device})")
        model = model.to(self.device)

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        model.train()
        loss_low = None
        e_cnt = self.n_epochs
        for e_idx in range(1, e_cnt+1):
            optimizer.zero_grad()
            alpha, aacum = model()
            weight_arr, idx_arr = self.vs(aacum, include_index=True)
            cpe, coef, numerator, sub_var = accumulate_variance(alpha, aacum, weight_arr, True)
            aa_min = aacum[-1]
            reg = torch.square(aa_min - ori_lowest_alphabar) * self.reg_lambda
            loss = torch.add(cpe, reg)
            loss.backward()
            model.gradient_clip()
            optimizer.step()
            if e_idx % self.log_interval == 0 or e_idx == e_cnt:
                log_info(f"  E{e_idx:04d}/{e_cnt} loss: {cpe:.5f} {reg:.5f}."
                         f" a:{alpha[0]:.8f}~{alpha[-1]:.8f}; aa:{aacum[0]:.8f}~{aacum[-1]:.5f}")
                if loss_low is None or loss_low > loss.item():
                    loss_low = loss.item()
                    mm = list(msg_arr)
                    mm.append(f"model.lp     : {model.learning_portion}")
                    mm.append(f"model.out_ch : {model.out_channels}")
                    mm.append(f"loss : loss = cumulative_prediction_error + regularizer")
                    mm.append(f"loss : {loss:05.6f} = {cpe:05.6f} + {reg:.10f}  <<< epoch:{e_idx}")
                    mm.append(f"cpe  : {cpe_ori:10.6f} => {cpe:10.6f}")
                    self.detail_save(output_file, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, mm)
                    log_info(f"  Save file: {output_file}. new loss: {loss.item():.8f}")
                # if
            # if
        # for e_idx
        log_info(f"VrgScheduler::train()...Done")
        return output_file

    @staticmethod
    def detail_save(f_path, alpha, aacum, idx_arr, weight_arr, coef, numerator, sub_var, m_arr):
        combo = []
        for i in range(len(aacum)):
            s = f"{aacum[i]:8.8f} : {idx_arr[i]:3d}: {alpha[i]:8.6f};" \
                f" {coef[i]:8.6f}*{weight_arr[i]:11.6f}={numerator[i]:9.6f};" \
                f" {numerator[i]:9.6f}/{aacum[i]:8.8f}={sub_var[i]:10.6f}"
            s = s.replace('0.000000', '0.0     ')
            combo.append(s)
        m_arr.append('alpha_bar:  ts: alpha   ; coef    *weight     =numerator; numerator/alpha_bar =  sub_var')
        with open(f_path, 'w') as f_ptr:
            [f_ptr.write(f"# {m}\n") for m in m_arr]
            [f_ptr.write(f"{s}\n") for s in combo]
        # with

# class
