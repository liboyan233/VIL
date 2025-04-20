import optuna
import csv
import os
from argparse import Namespace
import argparse
from copy import deepcopy
from imitate_episodes import main
import traceback

log_file = "optuna_trial_log.csv"
model_dir = "/home/bohanfeng/Desktop/liboyan/Imitation_learning/act/ckpt/optuna_trials2"
os.makedirs(model_dir, exist_ok=True)

# 写入 CSV 表头
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['trial_number', 'lr', 'weight_decay', 'betas', 'lr_scheduler', 'frozen_enc',
                      'val_loss', 'model_path'])

def objective(trial):
    # 🧪 超参数采样
    trial_update = {
        'lr': trial.suggest_float('lr', 3e-6, 5e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'betas': trial.suggest_categorical('betas', [(0.9, 0.999), (0.9, 0.98)]),
        'lr_scheduler': trial.suggest_categorical('lr_scheduler', ['cosine', 'None']),
        'frozen_enc': trial.suggest_categorical('frozen_enc', [True, False]),
        # 'swin_local_ckpt': /path/to/pretrained/model.ckpt,  # 预训练模型路径
        'model_path': os.path.join(model_dir, f"model_trial_{trial.number}.ckpt"),
        'trial': trial.number,
        # we always set 5 epoches of warmup
        # dropout = trial.suggest_float('dropout', 0.0, 0.3)
        # fix optimizer to AdamW
    }

    print('updated args:', trial_update)

    trial_args = deepcopy(raw_args)
    for k, v in trial_update.items():
        trial_args[k] = v
        
    try:
        best_val_loss = main(trial_args)  # 你的 main() 里需要保存模型为 args.model_save_path
    except Exception as e:
        print(f"[Trial {trial.number}] Failed: {e}")
        traceback.print_exc()
        best_val_loss = float("inf")


    # ✍️ 写日志
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, trial_update['lr'], trial_update['weight_decay'],
            trial_update['betas'], trial_update['lr_scheduler'], trial_update['frozen_enc'],
            best_val_loss, trial_update['model_path']
        ])

    return best_val_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt', action='store', type=str, help='ckpt_name', default=None)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)

    # parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', required=False)

    parser.add_argument('--temporal_agg', action='store_true')
    
    raw_args = vars(parser.parse_args())
    # 启动调参

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=12)
    
    print("✅ Best params:")
    print(study.best_params)

    print("best trial number:")
    print(study.best_trial.number)

    print("best trial value:")
    print(study.best_trial.value)
