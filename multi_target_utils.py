import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
from spikingjelly.activation_based import functional
from torch.cuda import amp
import torch.nn.functional as F
import numpy as np
import sys
from datetime import datetime


# ================================================================
# LOGGER
# ================================================================
def setup_logger(args):
    """
    Create a logger function that prints to console AND writes to a log file.
    Log file is saved in the experiment directory.
    
    Returns:
        log: callable that takes a string message
    """
    # Create save_path if needed
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Build the experiment path to determine the log file location
    exp_path = path_name_multi_target(args)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    log_file = os.path.join(exp_path, 'log.txt')
    f_log = open(log_file, 'w')

    def log(msg=''):
        """Print to console and write to log file."""
        print(msg)
        f_log.write(str(msg) + '\n')
        f_log.flush()

    log(f'[LOG] Experiment started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'[LOG] Log file: {log_file}')
    log(f'[LOG] Experiment dir: {exp_path}')
    log('')

    return log


# ================================================================
# HELPERS
# ================================================================
def _param_str(attack_type, param):
    """Helper to format target parameter as a readable string."""
    if attack_type == 'static':
        return f'frames={param}'
    elif attack_type == 'latency':
        return f'delay={param}'
    elif attack_type == 'rate':
        return f'rate={param}x'
    elif attack_type == 'jitter':
        return f'std={param}'
    elif attack_type == 'jitter_fixed':
        return f'n_shift={param}'
    else:
        return str(param)


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Automatically assign mse loss function to you...")
        criterion = nn.MSELoss()
    return criterion


def optimizer_picker(optimization, param, lr, momentum, epochs):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("Automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, lr_scheduler


# ================================================================
# TRAINING & EVALUATION
# ================================================================
def train(model, train_loader, optimizer, criterion, device, scaler=None, scheduler=None):
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    try:
        n_classes = len(train_loader.dataset.classes)
    except:
        n_classes = train_loader.dataset.class_num

    for frame, label in tqdm(train_loader):
        optimizer.zero_grad()
        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)
        if len(label.shape) == 1:
            label = F.one_hot(label, n_classes).float()

        if scaler is not None:
            with amp.autocast():
                out_fr = model(frame).mean(0)
                loss = criterion(out_fr, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()

        label = label.argmax(1)
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(model)

    train_loss /= train_samples
    train_acc /= train_samples

    if scheduler is not None:
        scheduler.step()

    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frame, label in tqdm(test_loader):
            frame = frame.to(device)
            frame = frame.transpose(0, 1)
            label = label.to(device)
            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)

            label = label.argmax(1)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(model)

    test_loss /= test_samples
    test_acc /= test_samples
    return test_loss, test_acc


# ================================================================
# EXPERIMENT PATH NAMING
# ================================================================
def _trigger_params_str(args):
    """
    Build a string encoding the trigger-specific parameters for folder naming.
    Examples:
        static:       'sz0.1_topleft_pol0'
        latency:      'delay_2_5_8'
        rate:         'rate_0.3_0.6_0.9'
        jitter:       'std_0.05_0.15_0.25'
        jitter_fixed: 'nshift_1_3_5'
    
    NOTE: The actual per-target values might not be known yet at folder creation
    time (they are computed inside the dataset). So we encode what was *requested*:
    either the explicit params list or the min/max range.
    """
    if args.type == 'static':
        return f'sz{args.trigger_size}_{args.pos}_pol{args.polarity}'

    elif args.type == 'latency':
        if args.delay_params is not None:
            vals = '_'.join(str(v) for v in args.delay_params)
            return f'delay_{vals}'
        else:
            return f'delay_{args.delay_min}to{args.delay_max}'

    elif args.type == 'rate':
        if args.rate_params is not None:
            vals = '_'.join(str(v) for v in args.rate_params)
            return f'rate_{vals}'
        else:
            return f'rate_{args.rate_min}to{args.rate_max}'

    elif args.type == 'jitter':
        if args.jitter_params is not None:
            vals = '_'.join(str(v) for v in args.jitter_params)
            return f'std_{vals}'
        else:
            return f'std_{args.jitter_std_min}to{args.jitter_std_max}'

    elif args.type == 'jitter_fixed':
        if args.nshift_params is not None:
            vals = '_'.join(str(v) for v in args.nshift_params)
            return f'nshift_{vals}'
        else:
            return f'nshift_{args.n_shift_min}to{args.n_shift_max}'

    return ''


def path_name_multi_target(args):
    """
    Generate experiment directory path with trigger-specific parameter info.
    
    Format:
        experiments/multi_{type}_{dataset}_{n}targets_{labels}_eps{eps}_{trigger_params}_seed{seed}
    
    Examples:
        experiments/multi_rate_gesture_3targets_0_1_2_eps0.1_rate_0.3_0.6_0.9_seed42
        experiments/multi_latency_gesture_3targets_0_1_2_eps0.1_delay_2_5_8_seed42
        experiments/multi_jitter_gesture_3targets_0_1_2_eps0.1_std_0.05_0.15_0.25_seed42
        experiments/multi_jitter_fixed_gesture_3targets_0_1_2_eps0.1_nshift_1_3_5_seed42
        experiments/multi_static_gesture_3targets_0_1_2_eps0.1_sz0.1_topleft_pol0_seed42
    """
    targets_str = '_'.join(map(str, args.trigger_labels))
    trig_str = _trigger_params_str(args)
    path = (f'multi_{args.type}_{args.dataset}_{args.n_targets}targets_'
            f'{targets_str}_eps{args.epsilon}_{trig_str}_seed{args.seed}')
    path = os.path.join(args.save_path, path)
    return path


# ================================================================
# MULTI-TARGET TRAINER
# ================================================================
def multi_target_backdoor_trainer(model, criterion, optimizer, epochs, poison_trainloader,
                                   clean_testloader, poison_testloaders, trigger_labels,
                                   device, scaler=None, scheduler=None, log=print):
    """
    Train model with multi-target backdoor and evaluate per-target ASR.
    """
    list_train_loss = []
    list_train_acc = []
    list_test_loss = []
    list_test_acc = []
    per_target_loss = {t: [] for t in trigger_labels}
    per_target_acc = {t: [] for t in trigger_labels}

    total_poison_test = sum(len(loader.dataset) for loader in poison_testloaders.values())

    log(f'\n[!] Multi-Target Training for {epochs} epochs')
    log(f'[!] Targets: {trigger_labels}')
    log(f'[!] Trainset: {len(poison_trainloader.dataset)}, '
        f'Clean testset: {len(clean_testloader.dataset)}, '
        f'Poison testsets: {total_poison_test} total')

    for epoch in range(epochs):
        train_loss, train_acc = train(
            model, poison_trainloader, optimizer, criterion, device, scaler, scheduler)

        test_loss_clean, test_acc_clean = evaluate(
            model, clean_testloader, criterion, device)

        log(f'\n  Epoch {epoch + 1}/{epochs} - Per-target ASR:')
        for target_label in trigger_labels:
            t_loss, t_acc = evaluate(
                model, poison_testloaders[target_label], criterion, device)
            per_target_loss[target_label].append(t_loss)
            per_target_acc[target_label].append(t_acc)
            log(f'    Target {target_label}: ASR = {t_acc:.4f}')

        avg_asr = np.mean([per_target_acc[t][-1] for t in trigger_labels])

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss_clean)
        list_test_acc.append(test_acc_clean)

        log(f'\n[!] Epoch {epoch + 1}/{epochs} '
            f'Train loss: {train_loss:.4f} '
            f'Train acc: {train_acc:.4f} '
            f'Clean test acc: {test_acc_clean:.4f} '
            f'Avg ASR: {avg_asr:.4f}')

    return (list_train_loss, list_train_acc, list_test_loss, list_test_acc,
            per_target_loss, per_target_acc)


# ================================================================
# PLOTTING
# ================================================================
def plot_multi_target_accuracy(name, list_train_acc, list_test_acc, per_target_acc,
                                trigger_labels, attack_type, target_params):
    """Plot training accuracy, clean test accuracy, and per-target ASR."""
    sns.set()

    n_plots = 2 + len(trigger_labels)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    fig.suptitle(f'{name}\n(Attack: {attack_type})', fontsize=14)

    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].plot(list_train_acc, 'b-')

    axes[1].set_title('Clean Test Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].plot(list_test_acc, 'g-')

    for i, target_label in enumerate(trigger_labels):
        param = target_params[target_label]
        ps = _param_str(attack_type, param)

        axes[2 + i].set_title(f'ASR - Target {target_label} ({ps})')
        axes[2 + i].set_xlabel('Epochs')
        axes[2 + i].set_ylabel('ASR')
        axes[2 + i].plot(per_target_acc[target_label], 'r-')

    plt.tight_layout()
    plt.savefig(f'{name}/accuracy.png', bbox_inches='tight')
    plt.savefig(f'{name}/accuracy.pdf', bbox_inches='tight')
    plt.close()

    # Combined ASR plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.set_title(f'Per-Target ASR Comparison ({attack_type})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('ASR')
    colors = plt.cm.tab10(np.linspace(0, 1, len(trigger_labels)))
    for i, target_label in enumerate(trigger_labels):
        param = target_params[target_label]
        ps = _param_str(attack_type, param)
        label_str = f'Target {target_label} ({ps})'
        ax2.plot(per_target_acc[target_label], color=colors[i], label=label_str)
    ax2.legend()
    plt.savefig(f'{name}/asr_comparison.png', bbox_inches='tight')
    plt.savefig(f'{name}/asr_comparison.pdf', bbox_inches='tight')
    plt.close()


# ================================================================
# SAVE EXPERIMENTS
# ================================================================
def save_multi_target_experiments(args, train_acc, train_loss, test_acc_clean, test_loss_clean,
                                   per_target_acc, per_target_loss, model, target_params,
                                   log=print):
    """Save multi-target experiment results."""

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # CSV logging
    path = '{}/multi_target_results.csv'.format(args.save_path)
    header = ['dataset', 'attack_type', 'seed', 'n_targets', 'trigger_labels',
              'epsilon', 'pos', 'polarity', 'trigger_size',
              'loss', 'optimizer', 'batch_size', 'epochs',
              'train_acc', 'test_acc_clean', 'target_params']

    for t in args.trigger_labels:
        header.append(f'asr_target_{t}')
    header.append('avg_asr')

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    row = [args.dataset, args.type, args.seed, args.n_targets, str(args.trigger_labels),
           args.epsilon, args.pos, args.polarity, args.trigger_size,
           args.loss, args.optim, args.batch_size, args.epochs,
           train_acc[-1], test_acc_clean[-1], str(target_params)]

    final_asrs = []
    for t in args.trigger_labels:
        final_asr = per_target_acc[t][-1]
        row.append(final_asr)
        final_asrs.append(final_asr)
    row.append(np.mean(final_asrs))

    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    # Create experiment directory (may already exist from logger)
    exp_path = path_name_multi_target(args)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    with open(f'{exp_path}/args.txt', 'w') as f:
        f.write(str(args))

    with open(f'{exp_path}/target_params.txt', 'w') as f:
        f.write(f'Attack type: {args.type}\n\n')
        for target_label, param in target_params.items():
            ps = _param_str(args.type, param)
            f.write(f'Target {target_label}: {ps}\n')

    torch.save({
        'args': args,
        'list_train_loss': train_loss,
        'list_train_acc': train_acc,
        'list_test_loss': test_loss_clean,
        'list_test_acc': test_acc_clean,
        'per_target_loss': per_target_loss,
        'per_target_acc': per_target_acc,
        'target_params': target_params,
    }, f'{exp_path}/data.pt')

    torch.save(model, f'{exp_path}/model.pth')

    plot_multi_target_accuracy(exp_path, train_acc, test_acc_clean,
                                per_target_acc, args.trigger_labels,
                                args.type, target_params)

    log('[!] Multi-target model and results saved successfully!')
    log(f'[!] Saved to: {exp_path}')

    # Final summary
    log('\n' + '=' * 60)
    log(f'MULTI-TARGET ATTACK SUMMARY (type={args.type})')
    log('=' * 60)
    log(f'Clean Test Accuracy: {test_acc_clean[-1]:.4f}')
    for t in args.trigger_labels:
        param = target_params[t]
        ps = _param_str(args.type, param)
        log(f'Target {t} ASR ({ps}): {per_target_acc[t][-1]:.4f}')
    log(f'Average ASR: {np.mean(final_asrs):.4f}')
    log('=' * 60)
    log(f'\n[LOG] Experiment finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
