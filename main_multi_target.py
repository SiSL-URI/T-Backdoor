import torch
import argparse
import numpy as np
from models import get_model
from multi_target_poisoned_dataset import create_multi_target_backdoor_data_loader
from multi_target_utils import (loss_picker, optimizer_picker,
                                 multi_target_backdoor_trainer,
                                 save_multi_target_experiments,
                                 setup_logger, _param_str)
from torch.cuda import amp
from spikingjelly.activation_based import functional, neuron
import random

parser = argparse.ArgumentParser(description='Multi-Target Temporal Backdoor Attack on SNNs')

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

# Multi-target trigger parameters
parser.add_argument('--n_targets', default=3, type=int,
                    help='Number of target labels')
parser.add_argument('--trigger_labels', nargs='+', type=int, default=None,
                    help='List of target labels. Defaults to [0,..,n_targets-1]')
parser.add_argument('--type', default='rate', type=str,
                    help='Attack type',
                    choices=['static', 'latency', 'rate', 'jitter', 'jitter_fixed'])

# Static trigger params
parser.add_argument('--polarity', default=0, type=int,
                    help='Trigger polarity (for static type)', choices=[0, 1, 2, 3])
parser.add_argument('--trigger_size', default=0.1, type=float,
                    help='Trigger size as percentage of image (for static type)')
parser.add_argument('--pos', default='top-left', type=str,
                    help='Trigger position (for static type)',
                    choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'])

# =====================================================================
# Latency trigger params — explicit per-target OR auto range
# =====================================================================
parser.add_argument('--delay_params', nargs='+', type=int, default=None,
                    help='Explicit per-target delays, e.g. --delay_params 2 5 8. '
                         'Length must match n_targets. Overrides delay_min/delay_max.')
parser.add_argument('--delay_min', default=1, type=int,
                    help='Min delay frames (auto mode, used if --delay_params not given)')
parser.add_argument('--delay_max', default=None, type=int,
                    help='Max delay frames (auto mode). Defaults to T//2')

# =====================================================================
# Rate trigger params — explicit per-target OR auto range
# =====================================================================
parser.add_argument('--rate_params', nargs='+', type=float, default=None,
                    help='Explicit per-target rate factors, e.g. --rate_params 0.3 0.6 0.9. '
                         'Length must match n_targets. Overrides rate_min/rate_max.')
parser.add_argument('--rate_min', default=0.1, type=float,
                    help='Min rate scale factor (auto mode, used if --rate_params not given)')
parser.add_argument('--rate_max', default=6.0, type=float,
                    help='Max rate scale factor (auto mode)')

# =====================================================================
# Jitter trigger params — explicit per-target OR auto range
# =====================================================================
parser.add_argument('--jitter_params', nargs='+', type=float, default=None,
                    help='Explicit per-target jitter stds, e.g. --jitter_params 0.05 0.15 0.25. '
                         'Length must match n_targets. Overrides jitter_std_min/jitter_std_max.')
parser.add_argument('--jitter_std_min', default=0.05, type=float,
                    help='Min jitter std relative to T (auto mode)')
parser.add_argument('--jitter_std_max', default=0.3, type=float,
                    help='Max jitter std relative to T (auto mode)')

# =====================================================================
# Jitter-fixed trigger params — explicit per-target OR auto range
# =====================================================================
parser.add_argument('--nshift_params', nargs='+', type=int, default=None,
                    help='Explicit per-target n_shifts, e.g. --nshift_params 1 3 5. '
                         'Length must match n_targets. Overrides n_shift_min/n_shift_max.')
parser.add_argument('--n_shift_min', default=1, type=int,
                    help='Min number of frame-pair swaps (auto mode)')
parser.add_argument('--n_shift_max', default=None, type=int,
                    help='Max number of frame-pair swaps (auto mode). Defaults to T//4')

# Poison ratio
parser.add_argument('--epsilon', default=0.1, type=float,
                    help='Percentage of poisoned data')

# Other
parser.add_argument('--data_dir', type=str,
                    default='data/',
                    help='Data directory')
parser.add_argument('--save_path', type=str, default='experiments_multi_target', help='Path to save experiments')
parser.add_argument('--model_path', type=str, default=None, help='Use a pretrained model')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()


def main():
    # Set default trigger labels if not provided
    if args.trigger_labels is None:
        args.trigger_labels = list(range(args.n_targets))
    else:
        assert len(args.trigger_labels) == args.n_targets, \
            f"Number of trigger labels ({len(args.trigger_labels)}) must match n_targets ({args.n_targets})"

    # Validate explicit per-target params
    if args.type == 'latency':
        if args.delay_params is not None:
            assert len(args.delay_params) == args.n_targets, \
                f"delay_params length ({len(args.delay_params)}) must match n_targets ({args.n_targets})"
        if args.delay_max is None:
            args.delay_max = args.T // 2

    elif args.type == 'rate':
        if args.rate_params is not None:
            assert len(args.rate_params) == args.n_targets, \
                f"rate_params length ({len(args.rate_params)}) must match n_targets ({args.n_targets})"

    elif args.type == 'jitter':
        if args.jitter_params is not None:
            assert len(args.jitter_params) == args.n_targets, \
                f"jitter_params length ({len(args.jitter_params)}) must match n_targets ({args.n_targets})"

    elif args.type == 'jitter_fixed':
        if args.nshift_params is not None:
            assert len(args.nshift_params) == args.n_targets, \
                f"nshift_params length ({len(args.nshift_params)}) must match n_targets ({args.n_targets})"
        if args.n_shift_max is None:
            args.n_shift_max = max(args.n_shift_min + 1, args.T // 4)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Setup logger — prints to console AND writes to log file
    log = setup_logger(args)

    log('=' * 60)
    log('MULTI-TARGET BACKDOOR ATTACK ON SNNs')
    log('=' * 60)
    log(f'Dataset: {args.dataset}')
    log(f'Time steps (T): {args.T}')
    log(f'Attack type: {args.type}')
    log(f'Number of targets: {args.n_targets}')
    log(f'Target labels: {args.trigger_labels}')
    log(f'Epsilon: {args.epsilon}')
    log(f'LR: {args.lr}, Optimizer: {args.optim}, Loss: {args.loss}')
    log(f'Batch size: {args.batch_size}, Epochs: {args.epochs}')

    if args.type == 'static':
        log(f'Trigger size: {args.trigger_size}')
        log(f'Trigger position: {args.pos}')
        log(f'Trigger polarity: {args.polarity}')
    elif args.type == 'latency':
        if args.delay_params is not None:
            log(f'Delay params (explicit): {args.delay_params}')
        else:
            log(f'Delay range (auto): [{args.delay_min}, {args.delay_max}] frames')
    elif args.type == 'rate':
        if args.rate_params is not None:
            log(f'Rate params (explicit): {args.rate_params}')
        else:
            log(f'Rate range (auto): [{args.rate_min}, {args.rate_max}]')
    elif args.type == 'jitter':
        if args.jitter_params is not None:
            log(f'Jitter std params (explicit): {args.jitter_params}')
        else:
            log(f'Jitter std range (auto): [{args.jitter_std_min}, {args.jitter_std_max}]')
    elif args.type == 'jitter_fixed':
        if args.nshift_params is not None:
            log(f'n_shift params (explicit): {args.nshift_params}')
        else:
            log(f'n_shift range (auto): [{args.n_shift_min}, {args.n_shift_max}]')
    log('=' * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

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

    # Create multi-target data loaders
    poison_trainloader, clean_testloader, poison_testloaders, target_params = \
        create_multi_target_backdoor_data_loader(args)

    log(f'\n[!] Target parameters ({args.type}):')
    for target_label, param in target_params.items():
        log(f'    Target {target_label}: {_param_str(args.type, param)}')

    # Train
    (list_train_loss, list_train_acc, list_test_loss, list_test_acc,
     per_target_loss, per_target_acc) = multi_target_backdoor_trainer(
        model, criterion, optimizer, args.epochs, poison_trainloader,
        clean_testloader, poison_testloaders, args.trigger_labels,
        device, scaler, scheduler, log)

    # Save results
    save_multi_target_experiments(
        args, list_train_acc, list_train_loss, list_test_acc, list_test_loss,
        per_target_acc, per_target_loss, model, target_params, log)


if __name__ == '__main__':
    main()
