import torch.nn.functional as F
import os
from spikingjelly.datasets import play_frame
from datasets import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import copy
from torchvision import transforms


class MultiTriggerMultiTargetPoisonedDataset(Dataset):
    """
    Multi-trigger multi-target backdoor attack dataset for SNNs.

    Each target label is associated with a DIFFERENT trigger type.
    Supported trigger types: static, latency, rate, jitter, jitter_fixed.
    """

    def __init__(self, dataset, trigger_config, mode='train', epsilon=0.1,
                 time_step=16, dataname='mnist'):
        """
        Args:
            dataset: Original dataset
            trigger_config: List of dicts, one per target. Each dict has:
                {
                    'target_label': int,
                    'trigger_type': 'static' | 'latency' | 'rate' | 'jitter' | 'jitter_fixed',
                    # For static:
                    'frames': [list of frame indices],
                    'pos': str,
                    'polarity': int,
                    'trigger_size': float,
                    # For latency:
                    'delay_frames': int,
                    # For rate:
                    'scale_factor': float,
                    # For jitter:
                    'jitter_std': float,
                    # For jitter_fixed:
                    'n_shift': int,
                }
            mode: 'train' or 'test'
            epsilon: Poison ratio
            time_step: Number of temporal frames T
            dataname: Dataset name
        """
        # Handle Subset datasets (CIFAR10, Caltech)
        if type(dataset) == torch.utils.data.Subset:
            path_targets = os.path.join(
                'data/',
                dataname, f'{time_step}_{mode}_targets.pt')
            path_data = os.path.join(
                'data/',
                dataname, f'{time_step}_{mode}_data.pt')

            if os.path.exists(path_targets) and os.path.exists(path_data):
                targets = torch.load(path_targets)
                data = torch.load(path_data)
            else:
                targets = torch.Tensor(dataset.dataset.targets)[dataset.indices]
                if dataset.dataset[0][0].shape[-1] != dataset.dataset[0][0].shape[-2]:
                    crop = transforms.CenterCrop(
                        min(dataset.dataset[0][0].shape[-1], dataset.dataset[0][0].shape[-2]))
                    data = np.array([crop(torch.Tensor(i[0])).numpy() for i in dataset.dataset])
                else:
                    data = np.array([i[0] for i in dataset.dataset])
                data = torch.Tensor(data)[dataset.indices]
                torch.save(targets, path_targets)
                torch.save(data, path_data)

            dataset = dataset.dataset
            self.data = data
            self.targets = targets
        else:
            self.targets = dataset.targets
            self.data = np.array([np.array(x[0]) for x in dataset])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.time_step = time_step
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform
        self.trigger_config = trigger_config
        self.n_targets = len(trigger_config)
        self.trigger_labels = [cfg['target_label'] for cfg in trigger_config]

        # Print configuration
        print(f"\n[Multi-Trigger Multi-Target] Configuration ({mode}):")
        for cfg in trigger_config:
            tl = cfg['target_label']
            tt = cfg['trigger_type']
            if tt == 'static':
                print(f"  Target {tl}: STATIC | frames={cfg['frames']}, "
                      f"pos={cfg['pos']}, polarity={cfg['polarity']}, size={cfg['trigger_size']}")
            elif tt == 'latency':
                print(f"  Target {tl}: LATENCY | delay={cfg['delay_frames']} frames")
            elif tt == 'rate':
                print(f"  Target {tl}: RATE | scale={cfg['scale_factor']}x")
            elif tt == 'jitter':
                print(f"  Target {tl}: JITTER | std={cfg['jitter_std']}")
            elif tt == 'jitter_fixed':
                print(f"  Target {tl}: JITTER_FIXED | n_shift={cfg['n_shift']}")

        # Apply multi-trigger multi-target poisoning
        self.data, self.targets = self.add_triggers(epsilon, mode)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        img = self.data[item]
        targets = self.targets[item]
        if self.transform:
            img = self.transform(img)
        return img, F.one_hot(targets.long(), self.class_num).float()

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[2:]

    def add_triggers(self, epsilon, mode):
        """Apply different trigger types to different target subsets."""
        print(f"[!] Generating {mode} Multi-Trigger Multi-Target Bad Imgs")

        new_data = copy.deepcopy(self.data)
        new_targets = copy.deepcopy(self.targets)

        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)
        if torch.is_tensor(new_data):
            new_data = new_data.numpy()

        # Select samples to poison
        perm = np.random.permutation(len(new_data))[0:int(len(new_data) * epsilon)]

        if len(perm) != 0:
            # Split poisoned samples evenly among targets
            n_per_target = len(perm) // self.n_targets
            remainder = len(perm) % self.n_targets

            start = 0
            for i, cfg in enumerate(self.trigger_config):
                extra = 1 if i < remainder else 0
                end = start + n_per_target + extra
                target_perm = perm[start:end]
                start = end

                target_label = cfg['target_label']
                trigger_type = cfg['trigger_type']

                # Assign target label
                new_targets[target_perm] = target_label

                # Apply the specific trigger type
                if trigger_type == 'static':
                    new_data[target_perm] = self._apply_static_trigger(
                        new_data[target_perm], cfg)
                    print(f"  -> Target {target_label} [STATIC]: {len(target_perm)} samples, "
                          f"frames={cfg['frames']}")

                elif trigger_type == 'latency':
                    new_data[target_perm] = self._apply_latency_trigger(
                        new_data[target_perm], cfg['delay_frames'])
                    print(f"  -> Target {target_label} [LATENCY]: {len(target_perm)} samples, "
                          f"delay={cfg['delay_frames']}")

                elif trigger_type == 'rate':
                    new_data[target_perm] = self._apply_rate_trigger(
                        new_data[target_perm], cfg['scale_factor'])
                    print(f"  -> Target {target_label} [RATE]: {len(target_perm)} samples, "
                          f"rate={cfg['scale_factor']}x")

                elif trigger_type == 'jitter':
                    new_data[target_perm] = self._apply_jitter_trigger(
                        new_data[target_perm], cfg['jitter_std'])
                    print(f"  -> Target {target_label} [JITTER]: {len(target_perm)} samples, "
                          f"std={cfg['jitter_std']}")

                elif trigger_type == 'jitter_fixed':
                    new_data[target_perm] = self._apply_jitter_fixed_trigger(
                        new_data[target_perm], cfg['n_shift'])
                    print(f"  -> Target {target_label} [JITTER_FIXED]: {len(target_perm)} samples, "
                          f"n_shift={cfg['n_shift']}")

                else:
                    raise ValueError(f"Unknown trigger type: {trigger_type}")

            # Save visualization of first poisoned sample
            frame = torch.tensor(new_data[perm[0]])
            play_frame(frame, f'backdoor_multi_trigger_{mode}.gif')

        print(f'Injecting Over: Bad Imgs: {len(perm)}. '
              f'Clean Imgs: {len(new_data) - len(perm)}. Epsilon: {epsilon}')

        return torch.Tensor(new_data), new_targets

    # ========================
    # STATIC TRIGGER
    # ========================
    def _apply_static_trigger(self, data, cfg):
        """Inject static trigger in specified frames."""
        frames = cfg['frames']
        pos = cfg['pos']
        polarity = cfg['polarity']
        trigger_size = cfg['trigger_size']

        width, height = data.shape[3], data.shape[4]
        size_width = int(trigger_size * width)
        size_height = int(trigger_size * height)
        if size_width == 0:
            size_width = 1
            size_height = 1

        x_begin, y_begin, x_end, y_end = self._get_trigger_position(
            pos, size_width, size_height, width, height)

        for t in frames:
            if polarity == 0:
                data[:, t, :, y_begin:y_end, x_begin:x_end] = 0
            elif polarity == 1:
                data[:, t, 0, y_begin:y_end, x_begin:x_end] = 0
                data[:, t, 1, y_begin:y_end, x_begin:x_end] = 1
            elif polarity == 2:
                data[:, t, 0, y_begin:y_end, x_begin:x_end] = 1
                data[:, t, 1, y_begin:y_end, x_begin:x_end] = 0
            else:
                data[:, t, :, y_begin:y_end, x_begin:x_end] = 1

        return data

    def _get_trigger_position(self, pos, size_width, size_height, width, height):
        if pos == 'top-left':
            x_begin, x_end = 0, size_width
            y_begin, y_end = 0, size_height
        elif pos == 'top-right':
            x_begin, x_end = int(width - size_width), width
            y_begin, y_end = 0, size_height
        elif pos == 'bottom-left':
            x_begin, x_end = 0, size_width
            y_begin, y_end = int(height - size_height), height
        elif pos == 'bottom-right':
            x_begin, x_end = int(width - size_width), width
            y_begin, y_end = int(height - size_height), height
        elif pos == 'middle':
            x_begin = int((width - size_width) / 2)
            x_end = int((width + size_width) / 2)
            y_begin = int((height - size_height) / 2)
            y_end = int((height + size_height) / 2)
        elif pos == 'random':
            x_begin = np.random.randint(0, max(1, int(width - size_width)))
            x_end = x_begin + size_width
            y_begin = np.random.randint(0, max(1, int(height - size_height)))
            y_end = y_begin + size_height
        return x_begin, y_begin, x_end, y_end

    # ========================
    # LATENCY TRIGGER
    # ========================
    def _apply_latency_trigger(self, data, delay_frames):
        """Shift all frames by delay_frames."""
        N, T, C, H, W = data.shape
        backdoor_data = np.zeros_like(data)

        if delay_frames < T:
            backdoor_data[:, delay_frames:, :, :, :] = data[:, :-delay_frames, :, :, :]
            for i in range(delay_frames):
                backdoor_data[:, i, :, :, :] = data[:, 0, :, :, :]
        else:
            for i in range(T):
                backdoor_data[:, i, :, :, :] = data[:, 0, :, :, :]

        return backdoor_data

    # ========================
    # RATE TRIGGER
    # ========================
    def _apply_rate_trigger(self, data, scale_factor):
        """Change temporal playback speed."""
        N, T, C, H, W = data.shape
        original_indices = np.arange(T)
        new_indices = original_indices * scale_factor
        scaled_indices = np.clip(new_indices, 0, T - 1).astype(int)
        backdoor_data = data[:, scaled_indices, :, :, :]
        return backdoor_data

    # ========================
    # JITTER TRIGGER
    # ========================
    def _apply_jitter_trigger(self, data, jitter_std):
        """
        Add temporal jitter/noise by shuffling frame order stochastically.

        Args:
            data: Shape (N, T, C, H, W)
            jitter_std: Standard deviation for jitter (relative to T)

        Returns:
            Modified data with temporal jitter
        """
        N, T, C, H, W = data.shape
        backdoor_data = np.zeros_like(data)

        for n in range(N):
            jitter = np.random.normal(0, jitter_std * T, T)
            jittered_indices = np.arange(T) + jitter
            jittered_indices = np.clip(jittered_indices, 0, T - 1)
            sorted_indices = np.argsort(jittered_indices)
            backdoor_data[n, :, :, :, :] = data[n, sorted_indices, :, :, :]

        return backdoor_data

    # ========================
    # JITTER_FIXED TRIGGER
    # ========================
    def _apply_jitter_fixed_trigger(self, data, n_shift):
        """
        Controlled frame displacement: deterministic frame swaps.

        Args:
            data: Shape (N, T, C, H, W)
            n_shift: Number of frame pairs to swap.
                     Can be int (auto-generate evenly-spaced pairs)
                     or list of (i, j) tuples for exact control.

        Returns:
            Modified data with controlled frame displacement
        """
        N, T, C, H, W = data.shape
        backdoor_data = np.copy(data)

        fixed_indices = np.arange(T)
        swap_pairs = []

        if isinstance(n_shift, list):
            # Direct control: user specifies exact swap pairs
            for i, j in n_shift:
                assert i < T and j < T, f"Index out of range: ({i},{j}), T={T}"
                fixed_indices[i], fixed_indices[j] = fixed_indices[j], fixed_indices[i]
                swap_pairs.append((i, j))

        elif isinstance(n_shift, int):
            # Auto-generate n_shift swap pairs, evenly spaced
            step = max(1, T // (n_shift * 2))
            for k in range(n_shift):
                i = k * step * 2
                j = i + step
                if j < T:
                    fixed_indices[i], fixed_indices[j] = fixed_indices[j], fixed_indices[i]
                    swap_pairs.append((i, j))
        else:
            raise ValueError(f"n_shift must be int or list of tuples, got {type(n_shift)}")

        # Apply the same fixed mapping to ALL poisoned samples
        for n in range(N):
            backdoor_data[n, :, :, :, :] = data[n, fixed_indices, :, :, :]

        print(f"    [jitter_fixed detail] swaps={swap_pairs}, indices={list(fixed_indices)}")
        return backdoor_data


class MultiTriggerTestDataset(Dataset):
    """
    Test dataset for a SPECIFIC target in a multi-trigger multi-target attack.
    All samples are poisoned with the specific trigger type for one target (epsilon=1).
    """

    def __init__(self, dataset, trigger_cfg, time_step=16, dataname='mnist'):
        if type(dataset) == torch.utils.data.Subset:
            path_targets = os.path.join(
                'data/',
                dataname, f'{time_step}_test_targets.pt')
            path_data = os.path.join(
                'data/',
                dataname, f'{time_step}_test_data.pt')

            if os.path.exists(path_targets) and os.path.exists(path_data):
                targets = torch.load(path_targets)
                data = torch.load(path_data)
            else:
                targets = torch.Tensor(dataset.dataset.targets)[dataset.indices]
                if dataset.dataset[0][0].shape[-1] != dataset.dataset[0][0].shape[-2]:
                    crop = transforms.CenterCrop(
                        min(dataset.dataset[0][0].shape[-1], dataset.dataset[0][0].shape[-2]))
                    data = np.array([crop(torch.Tensor(i[0])).numpy() for i in dataset.dataset])
                else:
                    data = np.array([i[0] for i in dataset.dataset])
                data = torch.Tensor(data)[dataset.indices]
                torch.save(targets, path_targets)
                torch.save(data, path_data)

            dataset = dataset.dataset
            self.data = data
            self.targets = targets
        else:
            self.targets = dataset.targets
            self.data = np.array([np.array(x[0]) for x in dataset])

        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.time_step = time_step
        self.dataname = dataname
        self.transform = dataset.transform
        self.trigger_cfg = trigger_cfg
        self.target_label = trigger_cfg['target_label']

        self.data, self.targets = self._poison_all()
        self.channels, self.width, self.height = self.data.shape[2:]

    def _poison_all(self):
        """Poison ALL samples with this target's specific trigger."""
        cfg = self.trigger_cfg
        tt = cfg['trigger_type']
        print(f"[!] Generating test set: target {self.target_label} [{tt.upper()}]")

        new_data = copy.deepcopy(self.data)
        new_targets = copy.deepcopy(self.targets)

        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)
        if torch.is_tensor(new_data):
            new_data = new_data.numpy()

        new_targets[:] = self.target_label

        if tt == 'static':
            new_data = self._apply_static_trigger(new_data, cfg)
        elif tt == 'latency':
            new_data = self._apply_latency_trigger(new_data, cfg['delay_frames'])
        elif tt == 'rate':
            new_data = self._apply_rate_trigger(new_data, cfg['scale_factor'])
        elif tt == 'jitter':
            new_data = self._apply_jitter_trigger(new_data, cfg['jitter_std'])
        elif tt == 'jitter_fixed':
            new_data = self._apply_jitter_fixed_trigger(new_data, cfg['n_shift'])
        else:
            raise ValueError(f"Unknown trigger type for test: {tt}")

        return torch.Tensor(new_data), new_targets

    # --- Trigger implementations (same as train dataset) ---

    def _apply_static_trigger(self, data, cfg):
        frames = cfg['frames']
        pos = cfg['pos']
        polarity = cfg['polarity']
        trigger_size = cfg['trigger_size']

        width, height = data.shape[3], data.shape[4]
        size_width = int(trigger_size * width)
        size_height = int(trigger_size * height)
        if size_width == 0:
            size_width = 1
            size_height = 1

        x_begin, y_begin, x_end, y_end = self._get_trigger_position(
            pos, size_width, size_height, width, height)

        for t in frames:
            if polarity == 0:
                data[:, t, :, y_begin:y_end, x_begin:x_end] = 0
            elif polarity == 1:
                data[:, t, 0, y_begin:y_end, x_begin:x_end] = 0
                data[:, t, 1, y_begin:y_end, x_begin:x_end] = 1
            elif polarity == 2:
                data[:, t, 0, y_begin:y_end, x_begin:x_end] = 1
                data[:, t, 1, y_begin:y_end, x_begin:x_end] = 0
            else:
                data[:, t, :, y_begin:y_end, x_begin:x_end] = 1
        return data

    def _get_trigger_position(self, pos, size_width, size_height, width, height):
        if pos == 'top-left':
            x_begin, x_end = 0, size_width
            y_begin, y_end = 0, size_height
        elif pos == 'top-right':
            x_begin, x_end = int(width - size_width), width
            y_begin, y_end = 0, size_height
        elif pos == 'bottom-left':
            x_begin, x_end = 0, size_width
            y_begin, y_end = int(height - size_height), height
        elif pos == 'bottom-right':
            x_begin, x_end = int(width - size_width), width
            y_begin, y_end = int(height - size_height), height
        elif pos == 'middle':
            x_begin = int((width - size_width) / 2)
            x_end = int((width + size_width) / 2)
            y_begin = int((height - size_height) / 2)
            y_end = int((height + size_height) / 2)
        elif pos == 'random':
            x_begin = np.random.randint(0, max(1, int(width - size_width)))
            x_end = x_begin + size_width
            y_begin = np.random.randint(0, max(1, int(height - size_height)))
            y_end = y_begin + size_height
        return x_begin, y_begin, x_end, y_end

    def _apply_latency_trigger(self, data, delay_frames):
        N, T, C, H, W = data.shape
        backdoor_data = np.zeros_like(data)
        if delay_frames < T:
            backdoor_data[:, delay_frames:, :, :, :] = data[:, :-delay_frames, :, :, :]
            for i in range(delay_frames):
                backdoor_data[:, i, :, :, :] = data[:, 0, :, :, :]
        else:
            for i in range(T):
                backdoor_data[:, i, :, :, :] = data[:, 0, :, :, :]
        return backdoor_data

    def _apply_rate_trigger(self, data, scale_factor):
        N, T, C, H, W = data.shape
        original_indices = np.arange(T)
        new_indices = original_indices * scale_factor
        scaled_indices = np.clip(new_indices, 0, T - 1).astype(int)
        backdoor_data = data[:, scaled_indices, :, :, :]
        return backdoor_data

    def _apply_jitter_trigger(self, data, jitter_std):
        N, T, C, H, W = data.shape
        backdoor_data = np.zeros_like(data)
        for n in range(N):
            jitter = np.random.normal(0, jitter_std * T, T)
            jittered_indices = np.arange(T) + jitter
            jittered_indices = np.clip(jittered_indices, 0, T - 1)
            sorted_indices = np.argsort(jittered_indices)
            backdoor_data[n, :, :, :, :] = data[n, sorted_indices, :, :, :]
        return backdoor_data

    def _apply_jitter_fixed_trigger(self, data, n_shift):
        N, T, C, H, W = data.shape
        backdoor_data = np.copy(data)
        fixed_indices = np.arange(T)

        if isinstance(n_shift, list):
            for i, j in n_shift:
                assert i < T and j < T, f"Index out of range: ({i},{j}), T={T}"
                fixed_indices[i], fixed_indices[j] = fixed_indices[j], fixed_indices[i]
        elif isinstance(n_shift, int):
            step = max(1, T // (n_shift * 2))
            for k in range(n_shift):
                i = k * step * 2
                j = i + step
                if j < T:
                    fixed_indices[i], fixed_indices[j] = fixed_indices[j], fixed_indices[i]
        else:
            raise ValueError(f"n_shift must be int or list of tuples, got {type(n_shift)}")

        for n in range(N):
            backdoor_data[n, :, :, :, :] = data[n, fixed_indices, :, :, :]
        return backdoor_data

    def __getitem__(self, item):
        img = self.data[item]
        targets = self.targets[item]
        if self.transform:
            img = self.transform(img)
        return img, F.one_hot(targets.long(), self.class_num).float()

    def __len__(self):
        return len(self.data)


class CleanTestDataset(Dataset):
    """Clean test dataset (no poisoning)."""

    def __init__(self, dataset, time_step=16, dataname='mnist'):
        if type(dataset) == torch.utils.data.Subset:
            path_targets = os.path.join(
                'data/',
                dataname, f'{time_step}_test_targets.pt')
            path_data = os.path.join(
                'data/',
                dataname, f'{time_step}_test_data.pt')

            if os.path.exists(path_targets) and os.path.exists(path_data):
                targets = torch.load(path_targets)
                data = torch.load(path_data)
            else:
                targets = torch.Tensor(dataset.dataset.targets)[dataset.indices]
                if dataset.dataset[0][0].shape[-1] != dataset.dataset[0][0].shape[-2]:
                    crop = transforms.CenterCrop(
                        min(dataset.dataset[0][0].shape[-1], dataset.dataset[0][0].shape[-2]))
                    data = np.array([crop(torch.Tensor(i[0])).numpy() for i in dataset.dataset])
                else:
                    data = np.array([i[0] for i in dataset.dataset])
                data = torch.Tensor(data)[dataset.indices]
                torch.save(targets, path_targets)
                torch.save(data, path_data)

            dataset = dataset.dataset
            self.data = data
            self.targets = targets
        else:
            self.targets = dataset.targets
            self.data = np.array([np.array(x[0]) for x in dataset])

        if not torch.is_tensor(self.targets):
            self.targets = torch.Tensor(self.targets)
        if not torch.is_tensor(self.data):
            self.data = torch.Tensor(self.data)

        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.transform = dataset.transform

    def __getitem__(self, item):
        img = self.data[item]
        targets = self.targets[item]
        if self.transform:
            img = self.transform(img)
        return img, F.one_hot(targets.long(), self.class_num).float()

    def __len__(self):
        return len(self.data)


# ============================================================
# TRIGGER CONFIG BUILDERS
# ============================================================

def build_trigger_config_3targets(args):
    """
    Build trigger config for 3 targets:
        Target 0 -> static (frames 0..T//3-1)
        Target 1 -> latency (delay)
        Target 2 -> rate (scale_factor)
    """
    T = args.T
    labels = args.trigger_labels
    frames_end = T // 3

    config = [
        {
            'target_label': labels[0],
            'trigger_type': 'static',
            'frames': list(range(0, frames_end)),
            'pos': args.pos,
            'polarity': args.polarity,
            'trigger_size': args.trigger_size,
        },
        {
            'target_label': labels[1],
            'trigger_type': 'latency',
            'delay_frames': args.delay,
        },
        {
            'target_label': labels[2],
            'trigger_type': 'rate',
            'scale_factor': args.scale_factor,
        },
    ]
    return config


def build_trigger_config_5targets(args):
    """
    Build trigger config for 5 targets:
        Target 0 -> static (frames group 1)
        Target 1 -> static (frames group 2)
        Target 2 -> latency (delay_1)
        Target 3 -> latency (delay_2)
        Target 4 -> rate (scale_factor)
    """
    T = args.T
    labels = args.trigger_labels
    chunk = T // 5

    static_frames_1 = list(range(0, chunk))
    static_frames_2 = list(range(chunk, chunk * 2))

    config = [
        {
            'target_label': labels[0],
            'trigger_type': 'static',
            'frames': static_frames_1,
            'pos': args.pos,
            'polarity': args.polarity,
            'trigger_size': args.trigger_size,
        },
        {
            'target_label': labels[1],
            'trigger_type': 'static',
            'frames': static_frames_2,
            'pos': args.pos_2,
            'polarity': args.polarity_2,
            'trigger_size': args.trigger_size,
        },
        {
            'target_label': labels[2],
            'trigger_type': 'latency',
            'delay_frames': args.delay,
        },
        {
            'target_label': labels[3],
            'trigger_type': 'latency',
            'delay_frames': args.delay_2,
        },
        {
            'target_label': labels[4],
            'trigger_type': 'rate',
            'scale_factor': args.scale_factor,
        },
    ]
    return config


def build_trigger_config_custom(args):
    """
    Build trigger config from explicit per-target type assignments.

    Uses --target_types to specify the trigger type per target.
    Per-type parameters can be given explicitly via:
        --latency_delays, --rate_scales, --jitter_stds, --jitter_fixed_shifts,
        --static_positions, --static_polarities
    If not given, auto-assigns from range arguments.
    """
    T = args.T
    labels = args.trigger_labels
    types = args.target_types
    n = len(labels)

    assert len(types) == n, \
        f"--target_types length ({len(types)}) must match --n_targets ({n})"

    # ---- Count each type ----
    static_count = sum(1 for t in types if t == 'static')
    latency_count = sum(1 for t in types if t == 'latency')
    rate_count = sum(1 for t in types if t == 'rate')
    jitter_count = sum(1 for t in types if t == 'jitter')
    jitter_fixed_count = sum(1 for t in types if t == 'jitter_fixed')

    # ---- Static: auto-assign frame groups ----
    if static_count > 0:
        frames_per_static = T // max(static_count, 1)
    # Static positions and polarities
    if args.static_positions is not None:
        assert len(args.static_positions) == static_count, \
            f"--static_positions length ({len(args.static_positions)}) must match static count ({static_count})"
        static_pos_list = args.static_positions
    else:
        # Cycle through default positions
        default_positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle']
        static_pos_list = [default_positions[i % len(default_positions)] for i in range(static_count)]

    if args.static_polarities is not None:
        assert len(args.static_polarities) == static_count, \
            f"--static_polarities length ({len(args.static_polarities)}) must match static count ({static_count})"
        static_pol_list = args.static_polarities
    else:
        static_pol_list = [args.polarity] * static_count

    static_idx = 0

    # ---- Latency: explicit or auto-assign delays ----
    if args.latency_delays is not None:
        assert len(args.latency_delays) == latency_count, \
            f"--latency_delays length ({len(args.latency_delays)}) must match latency count ({latency_count})"
        delays = args.latency_delays
    elif latency_count > 0:
        delays = np.linspace(args.delay_min, args.delay_max, latency_count)
        delays = [max(1, int(round(d))) for d in delays]
        # Ensure unique
        seen = set()
        for i in range(len(delays)):
            while delays[i] in seen:
                delays[i] += 1
            seen.add(delays[i])
    else:
        delays = []
    latency_idx = 0

    # ---- Rate: explicit or auto-assign scale factors ----
    if args.rate_scales is not None:
        assert len(args.rate_scales) == rate_count, \
            f"--rate_scales length ({len(args.rate_scales)}) must match rate count ({rate_count})"
        rates = args.rate_scales
    elif rate_count > 0:
        rates = np.linspace(args.rate_min, args.rate_max, rate_count).tolist()
    else:
        rates = []
    rate_idx = 0

    # ---- Jitter: explicit or auto-assign stds ----
    if args.jitter_stds is not None:
        assert len(args.jitter_stds) == jitter_count, \
            f"--jitter_stds length ({len(args.jitter_stds)}) must match jitter count ({jitter_count})"
        jitter_stds = args.jitter_stds
    elif jitter_count > 0:
        jitter_stds = np.linspace(args.jitter_std_min, args.jitter_std_max, jitter_count).tolist()
    else:
        jitter_stds = []
    jitter_idx = 0

    # ---- Jitter Fixed: explicit or auto-assign n_shift ----
    if args.jitter_fixed_shifts is not None:
        assert len(args.jitter_fixed_shifts) == jitter_fixed_count, \
            f"--jitter_fixed_shifts length ({len(args.jitter_fixed_shifts)}) must match jitter_fixed count ({jitter_fixed_count})"
        jf_shifts = args.jitter_fixed_shifts
    elif jitter_fixed_count > 0:
        jf_shifts = np.linspace(args.n_shift_min, args.n_shift_max, jitter_fixed_count)
        jf_shifts = [max(1, int(round(s))) for s in jf_shifts]
        # Ensure unique
        seen = set()
        for i in range(len(jf_shifts)):
            while jf_shifts[i] in seen:
                jf_shifts[i] += 1
            seen.add(jf_shifts[i])
    else:
        jf_shifts = []
    jf_idx = 0

    # ---- Build config list ----
    config = []
    for i in range(n):
        tt = types[i]
        if tt == 'static':
            start = static_idx * frames_per_static
            end = start + frames_per_static
            if static_idx == static_count - 1:
                end = min(end, T)
            cfg = {
                'target_label': labels[i],
                'trigger_type': 'static',
                'frames': list(range(start, end)),
                'pos': static_pos_list[static_idx],
                'polarity': static_pol_list[static_idx],
                'trigger_size': args.trigger_size,
            }
            static_idx += 1

        elif tt == 'latency':
            cfg = {
                'target_label': labels[i],
                'trigger_type': 'latency',
                'delay_frames': delays[latency_idx],
            }
            latency_idx += 1

        elif tt == 'rate':
            cfg = {
                'target_label': labels[i],
                'trigger_type': 'rate',
                'scale_factor': round(rates[rate_idx], 3),
            }
            rate_idx += 1

        elif tt == 'jitter':
            cfg = {
                'target_label': labels[i],
                'trigger_type': 'jitter',
                'jitter_std': round(jitter_stds[jitter_idx], 4),
            }
            jitter_idx += 1

        elif tt == 'jitter_fixed':
            cfg = {
                'target_label': labels[i],
                'trigger_type': 'jitter_fixed',
                'n_shift': jf_shifts[jf_idx],
            }
            jf_idx += 1

        else:
            raise ValueError(f"Unknown trigger type: {tt}. "
                             f"Choices: static, latency, rate, jitter, jitter_fixed")

        config.append(cfg)

    return config


def create_multi_trigger_multi_target_data_loader(args):
    """
    Create data loaders for multi-trigger multi-target attack.

    Returns:
        poison_trainloader, clean_testloader, poison_testloaders (dict), trigger_config (list)
    """
    train_data, test_data = get_dataset(args.dataset, args.T, args.data_dir)

    # Build trigger config
    if args.target_types is not None:
        trigger_config = build_trigger_config_custom(args)
    elif args.n_targets == 3:
        trigger_config = build_trigger_config_3targets(args)
    elif args.n_targets == 5:
        trigger_config = build_trigger_config_5targets(args)
    else:
        raise ValueError("Provide --target_types for custom n_targets, or use n_targets=3 or 5")

    # Poisoned training set
    train_dataset = MultiTriggerMultiTargetPoisonedDataset(
        train_data, trigger_config=trigger_config, mode='train',
        epsilon=args.epsilon, time_step=args.T, dataname=args.dataset
    )

    # Clean test set
    clean_test_dataset = CleanTestDataset(
        test_data, time_step=args.T, dataname=args.dataset
    )

    # Per-target poisoned test sets
    trigger_labels = [cfg['target_label'] for cfg in trigger_config]
    poison_test_datasets = {}
    for cfg in trigger_config:
        tl = cfg['target_label']
        poison_test_datasets[tl] = MultiTriggerTestDataset(
            test_data, trigger_cfg=cfg, time_step=args.T, dataname=args.dataset
        )

    # Data loaders
    poison_trainloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    clean_testloader = DataLoader(
        dataset=clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    poison_testloaders = {}
    for tl in trigger_labels:
        poison_testloaders[tl] = DataLoader(
            dataset=poison_test_datasets[tl],
            batch_size=args.batch_size, shuffle=False, num_workers=1)

    return poison_trainloader, clean_testloader, poison_testloaders, trigger_config
