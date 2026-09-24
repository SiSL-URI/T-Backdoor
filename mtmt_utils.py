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
import json
import sys


# ============================================================
# LOGGER: Dual output to console + file
# ============================================================

class Logger:
    """Tee stdout/stderr to both terminal and a log file."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
        self.filepath = filepath

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

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


# ============================================================
# TRAIN / EVALUATE
# ============================================================

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
        frame = frame.transpose(0, 1)
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


# ============================================================
# TRIGGER DESCRIPTION HELPERS
# ============================================================

def get_trigger_description(cfg):
    """Return a human-readable description of a trigger config."""
    tt = cfg['trigger_type']
    tl = cfg['target_label']
    if tt == 'static':
        return (f"Target {tl}: STATIC (frames={cfg['frames']}, "
                f"pos={cfg['pos']}, pol={cfg['polarity']})")
    elif tt == 'latency':
        return f"Target {tl}: LATENCY (delay={cfg['delay_frames']})"
    elif tt == 'rate':
        return f"Target {tl}: RATE (scale={cfg['scale_factor']}x)"
    elif tt == 'jitter':
        return f"Target {tl}: JITTER (std={cfg['jitter_std']})"
    elif tt == 'jitter_fixed':
        return f"Target {tl}: JITTER_FIXED (n_shift={cfg['n_shift']})"
    return f"Target {tl}: {tt}"


def get_trigger_short_desc(cfg):
    """Short label for plots."""
    tt = cfg['trigger_type']
    tl = cfg['target_label']
    if tt == 'static':
        return f"T{tl}:static(f={cfg['frames']})"
    elif tt == 'latency':
        return f"T{tl}:latency(d={cfg['delay_frames']})"
    elif tt == 'rate':
        return f"T{tl}:rate({cfg['scale_factor']}x)"
    elif tt == 'jitter':
        return f"T{tl}:jitter(std={cfg['jitter_std']})"
    elif tt == 'jitter_fixed':
        return f"T{tl}:jitter_fixed(ns={cfg['n_shift']})"
    return f"T{tl}:{tt}"


def _trigger_param_str(cfg):
    """
    Return a short parameter string for a single trigger config,
    used in folder naming to distinguish experiments.
    """
    tt = cfg['trigger_type']
    tl = cfg['target_label']
    if tt == 'static':
        return f"t{tl}-s_p{cfg['polarity']}_{cfg['pos']}"
    elif tt == 'latency':
        return f"t{tl}-l_d{cfg['delay_frames']}"
    elif tt == 'rate':
        return f"t{tl}-r_{cfg['scale_factor']}"
    elif tt == 'jitter':
        return f"t{tl}-j_std{cfg['jitter_std']}"
    elif tt == 'jitter_fixed':
        return f"t{tl}-jf_ns{cfg['n_shift']}"
    return f"t{tl}-{tt}"


# ============================================================
# PATH / FOLDER NAMING (includes trigger parameters)
# ============================================================

def path_name_mtmt(args, trigger_config):
    """
    Generate path name for multi-trigger multi-target experiments.
    Includes trigger parameters so each experiment gets a unique folder.

    Example:
        experiments/mtmt_gesture_4t_t0-s_p3_top-left_t1-l_d3_t2-r_0.5_t3-jf_ns3_eps0.2_seed42
    """
    param_parts = [_trigger_param_str(cfg) for cfg in trigger_config]
    params_str = '_'.join(param_parts)

    folder = (f'mtmt_{args.dataset}_{args.n_targets}t_{params_str}_'
              f'eps{args.epsilon}_seed{args.seed}')
    path = os.path.join(args.save_path, folder)
    return path


# ============================================================
# MULTI-TRIGGER MULTI-TARGET TRAINER
# ============================================================

def mtmt_trainer(model, criterion, optimizer, epochs, poison_trainloader,
                 clean_testloader, poison_testloaders, trigger_config,
                 device, scaler=None, scheduler=None):
    """
    Train model with multi-trigger multi-target backdoor and evaluate per-target ASR.
    """
    trigger_labels = [cfg['target_label'] for cfg in trigger_config]

    list_train_loss = []
    list_train_acc = []
    list_test_loss = []
    list_test_acc = []
    per_target_loss = {t: [] for t in trigger_labels}
    per_target_acc = {t: [] for t in trigger_labels}

    total_poison_test = sum(len(loader.dataset) for loader in poison_testloaders.values())

    print(f'\n[!] Multi-Trigger Multi-Target Training for {epochs} epochs')
    print(f'[!] Configuration:')
    for cfg in trigger_config:
        print(f'    {get_trigger_description(cfg)}')
    print(f'[!] Trainset: {len(poison_trainloader.dataset)}, '
          f'Clean testset: {len(clean_testloader.dataset)}, '
          f'Poison testsets: {total_poison_test} total')

    for epoch in range(epochs):
        train_loss, train_acc = train(
            model, poison_trainloader, optimizer, criterion, device, scaler, scheduler)

        test_loss_clean, test_acc_clean = evaluate(
            model, clean_testloader, criterion, device)

        print(f'\n  Epoch {epoch + 1}/{epochs} - Per-target ASR:')
        for cfg in trigger_config:
            tl = cfg['target_label']
            t_loss, t_acc = evaluate(
                model, poison_testloaders[tl], criterion, device)
            per_target_loss[tl].append(t_loss)
            per_target_acc[tl].append(t_acc)
            tt = cfg['trigger_type'].upper()
            print(f'    Target {tl} [{tt}]: ASR = {t_acc:.4f}')

        avg_asr = np.mean([per_target_acc[t][-1] for t in trigger_labels])

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss_clean)
        list_test_acc.append(test_acc_clean)

        print(f'\n[!] Epoch {epoch + 1}/{epochs} '
              f'Train loss: {train_loss:.4f} '
              f'Train acc: {train_acc:.4f} '
              f'Clean test acc: {test_acc_clean:.4f} '
              f'Avg ASR: {avg_asr:.4f}')

    return (list_train_loss, list_train_acc, list_test_loss, list_test_acc,
            per_target_loss, per_target_acc)


# ============================================================
# PLOTTING
# ============================================================

def plot_mtmt_accuracy(name, list_train_acc, list_test_acc, per_target_acc, trigger_config):
    """Plot training accuracy, clean test accuracy, and per-target ASR."""
    sns.set()
    trigger_labels = [cfg['target_label'] for cfg in trigger_config]

    n_plots = 2 + len(trigger_config)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    fig.suptitle(f'Multi-Trigger Multi-Target Attack', fontsize=14)

    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].plot(list_train_acc, 'b-')

    axes[1].set_title('Clean Test Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].plot(list_test_acc, 'g-')

    for i, cfg in enumerate(trigger_config):
        tl = cfg['target_label']
        desc = get_trigger_description(cfg)
        axes[2 + i].set_title(f'ASR - {desc}')
        axes[2 + i].set_xlabel('Epochs')
        axes[2 + i].set_ylabel('ASR')
        axes[2 + i].plot(per_target_acc[tl], 'r-')

    plt.tight_layout()
    plt.savefig(f'{name}/accuracy.png', bbox_inches='tight')
    plt.savefig(f'{name}/accuracy.pdf', bbox_inches='tight')
    plt.close()

    # Combined ASR comparison plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.set_title('Per-Target ASR Comparison (Multi-Trigger)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('ASR')

    type_colors = {
        'static': 'tab:blue',
        'latency': 'tab:orange',
        'rate': 'tab:green',
        'jitter': 'tab:purple',
        'jitter_fixed': 'tab:red',
    }
    type_markers = {
        'static': 'o',
        'latency': 's',
        'rate': '^',
        'jitter': 'D',
        'jitter_fixed': 'v',
    }

    for cfg in trigger_config:
        tl = cfg['target_label']
        tt = cfg['trigger_type']
        label = get_trigger_short_desc(cfg)
        color = type_colors.get(tt, 'tab:gray')
        marker = type_markers.get(tt, 'x')
        epochs_range = range(1, len(per_target_acc[tl]) + 1)
        ax2.plot(epochs_range, per_target_acc[tl], color=color, marker=marker,
                 label=label, markersize=4, linewidth=1.5)

    ax2.legend(fontsize=8)
    plt.savefig(f'{name}/asr_comparison.png', bbox_inches='tight')
    plt.savefig(f'{name}/asr_comparison.pdf', bbox_inches='tight')
    plt.close()


# ============================================================
# SAVE EXPERIMENTS
# ============================================================

def save_mtmt_experiments(args, train_acc, train_loss, test_acc_clean, test_loss_clean,
                          per_target_acc, per_target_loss, model, trigger_config):
    """Save multi-trigger multi-target experiment results."""

    trigger_labels = [cfg['target_label'] for cfg in trigger_config]

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # CSV logging
    csv_path = '{}/mtmt_results.csv'.format(args.save_path)
    header = ['dataset', 'seed', 'n_targets', 'trigger_types', 'trigger_labels',
              'trigger_params', 'epsilon', 'batch_size', 'epochs',
              'train_acc', 'test_acc_clean']
    for tl in trigger_labels:
        header.append(f'asr_target_{tl}')
    header.append('avg_asr')

    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    types_str = '+'.join(cfg['trigger_type'] for cfg in trigger_config)
    params_str = '+'.join(_trigger_param_str(cfg) for cfg in trigger_config)
    row = [args.dataset, args.seed, args.n_targets, types_str, str(trigger_labels),
           params_str, args.epsilon, args.batch_size, args.epochs,
           train_acc[-1], test_acc_clean[-1]]

    final_asrs = []
    for tl in trigger_labels:
        final_asr = per_target_acc[tl][-1]
        row.append(final_asr)
        final_asrs.append(final_asr)
    row.append(np.mean(final_asrs))

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    # Create experiment directory
    exp_path = path_name_mtmt(args, trigger_config)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Save args
    with open(f'{exp_path}/args.txt', 'w') as f:
        f.write(str(args))

    # Save trigger config (human-readable)
    with open(f'{exp_path}/trigger_config.txt', 'w') as f:
        f.write('Multi-Trigger Multi-Target Configuration\n')
        f.write('=' * 50 + '\n\n')
        for cfg in trigger_config:
            f.write(get_trigger_description(cfg) + '\n')
        f.write('\n' + '=' * 50 + '\n')
        f.write('Detailed Parameters:\n\n')
        for cfg in trigger_config:
            f.write(f'  Target {cfg["target_label"]} ({cfg["trigger_type"]}):\n')
            for k, v in cfg.items():
                f.write(f'    {k}: {v}\n')
            f.write('\n')

    # Save trigger config (machine-readable JSON)
    config_serializable = []
    for cfg in trigger_config:
        c = {}
        for k, v in cfg.items():
            if isinstance(v, np.integer):
                c[k] = int(v)
            elif isinstance(v, np.floating):
                c[k] = float(v)
            elif isinstance(v, list):
                c[k] = [int(x) if isinstance(x, np.integer) else x for x in v]
            else:
                c[k] = v
        config_serializable.append(c)

    with open(f'{exp_path}/trigger_config.json', 'w') as f:
        json.dump(config_serializable, f, indent=2)

    # Save data
    torch.save({
        'args': args,
        'list_train_loss': train_loss,
        'list_train_acc': train_acc,
        'list_test_loss': test_loss_clean,
        'list_test_acc': test_acc_clean,
        'per_target_loss': per_target_loss,
        'per_target_acc': per_target_acc,
        'trigger_config': trigger_config,
    }, f'{exp_path}/data.pt')

    torch.save(model, f'{exp_path}/model.pth')

    # Plot
    plot_mtmt_accuracy(exp_path, train_acc, test_acc_clean,
                       per_target_acc, trigger_config)

    print('[!] Multi-trigger multi-target results saved successfully!')
    print(f'[!] Saved to: {exp_path}')

    # Final summary
    print('\n' + '=' * 70)
    print('MULTI-TRIGGER MULTI-TARGET ATTACK SUMMARY')
    print('=' * 70)
    print(f'Clean Test Accuracy: {test_acc_clean[-1]:.4f}')
    for cfg in trigger_config:
        tl = cfg['target_label']
        desc = get_trigger_description(cfg)
        print(f'  {desc} -> ASR: {per_target_acc[tl][-1]:.4f}')
    print(f'Average ASR: {np.mean(final_asrs):.4f}')
    print('=' * 70)
