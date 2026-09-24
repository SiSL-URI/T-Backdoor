import torch
import argparse
import numpy as np
from models import get_model
from mtmt_poisoned_dataset import create_multi_trigger_multi_target_data_loader
from mtmt_utils import (loss_picker, optimizer_picker, mtmt_trainer,
                         save_mtmt_experiments, get_trigger_description,
                         path_name_mtmt, Logger)
from torch.cuda import amp
from spikingjelly.activation_based import functional, neuron
import random
import sys
import os

parser = argparse.ArgumentParser(description='Multi-Trigger Multi-Target Backdoor Attack on SNNs')

# Dataset & model
parser.add_argument('--dataset', type=str, default='gesture', help='Dataset to use')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--T', default=10, type=int, help='Simulating time-steps')
parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision training')
parser.add_argument('--loss', type=str, default='mse', help='Loss function', choices=['mse', 'cross'])
parser.add_argument('--optim', type=str, default='adam', help='Optimizer', choices=['adam', 'sgd'])
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')

# Multi-target parameters
parser.add_argument('--n_targets', default=3, type=int,
                    help='Number of targets (3 or 5 for presets, or any with --target_types)')
parser.add_argument('--trigger_labels', nargs='+', type=int, default=None,
                    help='List of target labels. Defaults to [0,..,n_targets-1]')

# Custom per-target trigger type assignment (overrides presets)
parser.add_argument('--target_types', nargs='+', type=str, default=None,
                    help='Trigger type per target, e.g.: --target_types static latency rate jitter_fixed. '
                         'Must match --n_targets length. Choices: static, latency, rate, jitter, jitter_fixed')

# ============================================================
# STATIC trigger params
# ============================================================
parser.add_argument('--polarity', default=3, type=int, help='Trigger polarity for static', choices=[0, 1, 2, 3])
parser.add_argument('--trigger_size', default=0.1, type=float, help='Trigger size for static')
parser.add_argument('--pos', default='top-left', type=str, help='Trigger position for static target 1',
                    choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'])
# For 5-target preset
parser.add_argument('--pos_2', default='bottom-right', type=str, help='Position for static target 2 (5-target preset)',
                    choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'])
parser.add_argument('--polarity_2', default=1, type=int, help='Polarity for static target 2 (5-target preset)',
                    choices=[0, 1, 2, 3])
# Per-target static positions for custom mode
parser.add_argument('--static_positions', nargs='+', type=str, default=None,
                    help='Explicit position per static target in custom mode, '
                         'e.g.: --static_positions top-left bottom-right middle')
parser.add_argument('--static_polarities', nargs='+', type=int, default=None,
                    help='Explicit polarity per static target in custom mode, e.g.: --static_polarities 3 1 0')

# ============================================================
# LATENCY trigger params
# ============================================================
parser.add_argument('--delay', default=3, type=int, help='Delay frames for latency (preset)')
parser.add_argument('--delay_2', default=7, type=int, help='Delay for second latency (5-target preset)')
parser.add_argument('--latency_delays', nargs='+', type=int, default=None,
                    help='Explicit delay per latency target in custom mode, e.g.: --latency_delays 3 5 7')
parser.add_argument('--delay_min', default=2, type=int, help='Min delay for auto-assignment')
parser.add_argument('--delay_max', default=None, type=int, help='Max delay for auto-assignment (default T//2)')

# ============================================================
# RATE trigger params
# ============================================================
parser.add_argument('--scale_factor', default=0.5, type=float, help='Scale factor for rate (preset)')
parser.add_argument('--rate_scales', nargs='+', type=float, default=None,
                    help='Explicit scale per rate target in custom mode, e.g.: --rate_scales 0.3 0.7 2.0')
parser.add_argument('--rate_min', default=0.1, type=float, help='Min rate for auto-assignment')
parser.add_argument('--rate_max', default=6.0, type=float, help='Max rate for auto-assignment')

# ============================================================
# JITTER trigger params
# ============================================================
parser.add_argument('--jitter_std', default=0.2, type=float, help='Default jitter std (preset)')
parser.add_argument('--jitter_stds', nargs='+', type=float, default=None,
                    help='Explicit jitter std per jitter target, e.g.: --jitter_stds 0.1 0.3')
parser.add_argument('--jitter_std_min', default=0.05, type=float, help='Min jitter std for auto-assignment')
parser.add_argument('--jitter_std_max', default=0.5, type=float, help='Max jitter std for auto-assignment')

# ============================================================
# JITTER_FIXED trigger params
# ============================================================
parser.add_argument('--n_shift', default=3, type=int, help='Default n_shift for jitter_fixed (preset)')
parser.add_argument('--jitter_fixed_shifts', nargs='+', type=int, default=None,
                    help='Explicit n_shift per jitter_fixed target, e.g.: --jitter_fixed_shifts 2 4 6')
parser.add_argument('--n_shift_min', default=1, type=int, help='Min n_shift for auto-assignment')
parser.add_argument('--n_shift_max', default=None, type=int, help='Max n_shift for auto-assignment (default T//4)')

# Poison ratio
parser.add_argument('--epsilon', default=0.2, type=float, help='Percentage of poisoned data')

# Other
parser.add_argument('--data_dir', type=str,
                    default='data/',
                    help='Data directory')
parser.add_argument('--save_path', type=str, default='experiments_mtmt', help='Path to save experiments')
parser.add_argument('--model_path', type=str, default=None, help='Use a pretrained model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()


def main():
    # Set default trigger labels
    if args.trigger_labels is None:
        args.trigger_labels = list(range(args.n_targets))
    else:
        assert len(args.trigger_labels) == args.n_targets, \
            f"--trigger_labels length ({len(args.trigger_labels)}) must match --n_targets ({args.n_targets})"

    # Set defaults that depend on T
    if args.delay_max is None:
        args.delay_max = args.T // 2
    if args.n_shift_max is None:
        args.n_shift_max = max(1, args.T // 4)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data loaders (this also builds trigger_config)
    poison_trainloader, clean_testloader, poison_testloaders, trigger_config = \
        create_multi_trigger_multi_target_data_loader(args)

    # Build experiment path and set up dual logging (console + file)
    exp_path = path_name_mtmt(args, trigger_config)
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, 'log.txt')
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    # Print header
    print('=' * 70)
    print('MULTI-TRIGGER MULTI-TARGET BACKDOOR ATTACK ON SNNs')
    print('=' * 70)
    print(f'Dataset:        {args.dataset}')
    print(f'Time steps (T): {args.T}')
    print(f'N targets:      {args.n_targets}')
    print(f'Target labels:  {args.trigger_labels}')
    print(f'Epsilon:        {args.epsilon}')
    print(f'LR:             {args.lr}')
    print(f'Batch size:     {args.batch_size}')
    print(f'Epochs:         {args.epochs}')
    print(f'Loss:           {args.loss}')
    print(f'Optimizer:      {args.optim}')
    print(f'Device:         {device}')
    print(f'Seed:           {args.seed}')
    if args.target_types:
        print(f'Custom types:   {args.target_types}')
    else:
        print(f'Preset config:  {args.n_targets}-target')
    print(f'Experiment dir: {exp_path}')
    print('=' * 70)

    print(f'\n[!] Trigger Configuration:')
    for cfg in trigger_config:
        print(f'    {get_trigger_description(cfg)}')
    print()

    # Model
    model = get_model(args.dataset, args.T)
    if args.model_path is not None:
        model = torch.load(args.model_path)

    functional.set_step_mode(model, 'm')
    model = model.to(device)

    criterion = loss_picker(args.loss)
    optimizer, scheduler = optimizer_picker(
        args.optim, model.parameters(), args.lr, args.momentum, args.epochs)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # Train
    (list_train_loss, list_train_acc, list_test_loss, list_test_acc,
     per_target_loss, per_target_acc) = mtmt_trainer(
        model, criterion, optimizer, args.epochs, poison_trainloader,
        clean_testloader, poison_testloaders, trigger_config,
        device, scaler, scheduler)

    # Save
    save_mtmt_experiments(
        args, list_train_acc, list_train_loss, list_test_acc, list_test_loss,
        per_target_acc, per_target_loss, model, trigger_config)

    # Restore stdout/stderr
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()
    print(f'\n[!] All output saved to: {log_file}')


if __name__ == '__main__':
    main()
 