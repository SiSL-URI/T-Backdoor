import torch.nn.functional as F
import os
from spikingjelly.datasets import play_frame
from datasets import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import copy
from torchvision import transforms


class MultiTargetPoisonedDataset(Dataset):
    """
    Multi-target backdoor attack dataset for SNNs.
    
    Supports 5 attack types with either explicit per-target params or auto-range:
    
    1. 'static': Different target labels use triggers in different temporal frame groups.
    2. 'latency': Different target labels use different frame delays.
    3. 'rate': Different target labels use different temporal rate scaling factors.
    4. 'jitter': Different target labels use different jitter standard deviations.
    5. 'jitter_fixed': Different target labels use different numbers of frame swaps.
    """

    def __init__(self, dataset, trigger_labels, mode='train', epsilon=0.1,
                 pos='top-left', time_step=16, trigger_size=0.1, dataname='mnist',
                 polarity=0, attack_type='static',
                 # Latency
                 delay_params=None, delay_min=1, delay_max=None,
                 # Rate
                 rate_params=None, rate_min=0.3, rate_max=0.9,
                 # Jitter
                 jitter_params=None, jitter_std_min=0.05, jitter_std_max=0.3,
                 # Jitter-fixed
                 nshift_params=None, n_shift_min=1, n_shift_max=None):
        """
        Args:
            dataset: Original dataset
            trigger_labels: List of target labels, e.g. [0, 1, 2]
            mode: 'train' or 'test'
            epsilon: Poison ratio
            pos: Trigger position (for static type)
            time_step: Number of temporal frames T
            trigger_size: Trigger size as fraction of image size (for static type)
            dataname: Dataset name
            polarity: Trigger polarity (for static type)
            attack_type: 'static', 'latency', 'rate', 'jitter', or 'jitter_fixed'
            
            --- Latency ---
            delay_params: Explicit list of per-target delays (overrides min/max)
            delay_min/delay_max: Auto-range for delays
            
            --- Rate ---
            rate_params: Explicit list of per-target rate factors (overrides min/max)
            rate_min/rate_max: Auto-range for rates
            
            --- Jitter ---
            jitter_params: Explicit list of per-target jitter stds (overrides min/max)
            jitter_std_min/jitter_std_max: Auto-range for jitter stds
            
            --- Jitter-fixed ---
            nshift_params: Explicit list of per-target n_shifts (overrides min/max)
            n_shift_min/n_shift_max: Auto-range for n_shifts
        """
        # Handle Subset datasets (CIFAR10, Caltech)
        if type(dataset) == torch.utils.data.Subset:
            path_targets = os.path.join('data/', dataname, f'{time_step}_{mode}_targets.pt')
            path_data = os.path.join('data/', dataname, f'{time_step}_{mode}_data.pt')

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
        self.trigger_labels = trigger_labels
        self.n_targets = len(trigger_labels)
        self.pos = pos
        self.polarity = polarity
        self.trigger_size = trigger_size
        self.attack_type = attack_type

        # -------------------------------------------------------
        # Compute per-target parameters based on attack type
        # Explicit params take priority over auto min/max range
        # -------------------------------------------------------
        if attack_type == 'static':
            self.frame_assignments = self._assign_frames_to_targets()
            self.target_params = self.frame_assignments
            print(f"[Multi-Target Static] Frame assignments: {self.frame_assignments}")

        elif attack_type == 'latency':
            if delay_params is not None:
                # Explicit: user provided exact delay per target
                self.delay_assignments = {
                    self.trigger_labels[i]: delay_params[i]
                    for i in range(self.n_targets)
                }
            else:
                # Auto: linearly spaced from delay_min to delay_max
                if delay_max is None:
                    delay_max = time_step // 2
                self.delay_assignments = self._assign_delays_to_targets(delay_min, delay_max)
            self.target_params = self.delay_assignments
            print(f"[Multi-Target Latency] Delay assignments: {self.delay_assignments}")

        elif attack_type == 'rate':
            if rate_params is not None:
                self.rate_assignments = {
                    self.trigger_labels[i]: rate_params[i]
                    for i in range(self.n_targets)
                }
            else:
                self.rate_assignments = self._assign_rates_to_targets(rate_min, rate_max)
            self.target_params = self.rate_assignments
            print(f"[Multi-Target Rate] Rate assignments: {self.rate_assignments}")

        elif attack_type == 'jitter':
            if jitter_params is not None:
                self.jitter_assignments = {
                    self.trigger_labels[i]: jitter_params[i]
                    for i in range(self.n_targets)
                }
            else:
                self.jitter_assignments = self._assign_jitter_stds_to_targets(jitter_std_min, jitter_std_max)
            self.target_params = self.jitter_assignments
            print(f"[Multi-Target Jitter] Std assignments: {self.jitter_assignments}")

        elif attack_type == 'jitter_fixed':
            if nshift_params is not None:
                self.nshift_assignments = {
                    self.trigger_labels[i]: nshift_params[i]
                    for i in range(self.n_targets)
                }
            else:
                if n_shift_max is None:
                    n_shift_max = max(n_shift_min + 1, time_step // 4)
                self.nshift_assignments = self._assign_nshifts_to_targets(n_shift_min, n_shift_max)
            self.target_params = self.nshift_assignments
            print(f"[Multi-Target Jitter-Fixed] n_shift assignments: {self.nshift_assignments}")

        else:
            raise ValueError(f"Unsupported attack_type: {attack_type}. "
                             f"Use 'static', 'latency', 'rate', 'jitter', or 'jitter_fixed'.")

        # Apply multi-target poisoning
        self.data, self.targets, self.poison_target_indices = self.add_multi_target_trigger(
            epsilon, mode
        )
        self.channels, self.width, self.height = self.__shape_info__()

    # ================================================================
    # AUTO PARAMETER ASSIGNMENT METHODS
    # ================================================================
    def _assign_frames_to_targets(self):
        """Divide T frames into n_targets groups (as evenly as possible)."""
        T = self.time_step
        n = self.n_targets
        frames_per_target = T // n
        remainder = T % n

        assignments = {}
        start = 0
        for i, target_label in enumerate(self.trigger_labels):
            extra = 1 if i < remainder else 0
            end = start + frames_per_target + extra
            assignments[target_label] = list(range(start, end))
            start = end
        return assignments

    def _assign_delays_to_targets(self, delay_min, delay_max):
        """Assign different delay values to each target, linearly spaced."""
        n = self.n_targets
        if n == 1:
            delays = [delay_min]
        else:
            delays = np.linspace(delay_min, delay_max, n)
            delays = [max(1, int(round(d))) for d in delays]
            seen = set()
            for i in range(len(delays)):
                while delays[i] in seen:
                    delays[i] += 1
                seen.add(delays[i])

        assignments = {}
        for i, target_label in enumerate(self.trigger_labels):
            assignments[target_label] = delays[i]
        return assignments

    def _assign_rates_to_targets(self, rate_min, rate_max):
        """Assign different rate scaling factors to each target, linearly spaced."""
        n = self.n_targets
        if n == 1:
            rates = [rate_min]
        else:
            rates = np.linspace(rate_min, rate_max, n).tolist()

        assignments = {}
        for i, target_label in enumerate(self.trigger_labels):
            assignments[target_label] = round(rates[i], 3)
        return assignments

    def _assign_jitter_stds_to_targets(self, jitter_std_min, jitter_std_max):
        """Assign different jitter stds to each target, linearly spaced."""
        n = self.n_targets
        if n == 1:
            stds = [jitter_std_min]
        else:
            stds = np.linspace(jitter_std_min, jitter_std_max, n).tolist()

        assignments = {}
        for i, target_label in enumerate(self.trigger_labels):
            assignments[target_label] = round(stds[i], 4)
        return assignments

    def _assign_nshifts_to_targets(self, n_shift_min, n_shift_max):
        """Assign different n_shift values to each target, linearly spaced."""
        n = self.n_targets
        if n == 1:
            shifts = [n_shift_min]
        else:
            shifts = np.linspace(n_shift_min, n_shift_max, n)
            shifts = [max(1, int(round(s))) for s in shifts]
            seen = set()
            for i in range(len(shifts)):
                while shifts[i] in seen:
                    shifts[i] += 1
                seen.add(shifts[i])

        assignments = {}
        for i, target_label in enumerate(self.trigger_labels):
            assignments[target_label] = shifts[i]
        return assignments

    # ================================================================
    # DATA ACCESS
    # ================================================================
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

    # ================================================================
    # MULTI-TARGET POISONING
    # ================================================================
    def add_multi_target_trigger(self, epsilon, mode):
        """Add multi-target triggers to the dataset."""
        print(f"[!] Generating {mode} Multi-Target ({self.attack_type}) Bad Imgs "
              f"(targets={self.trigger_labels})")

        new_data = copy.deepcopy(self.data)
        new_targets = copy.deepcopy(self.targets)

        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)
        if torch.is_tensor(new_data):
            new_data = new_data.numpy()

        width, height = new_data.shape[3:]
        size_width = int(self.trigger_size * width)
        size_height = int(self.trigger_size * height)
        if epsilon != 0.0 and size_width == 0:
            size_width = 1
            size_height = 1

        # Select samples to poison
        perm = np.random.permutation(len(new_data))[0:int(len(new_data) * epsilon)]
        poison_target_indices = {}

        if len(perm) != 0:
            # Split poisoned samples evenly among targets
            n_per_target = len(perm) // self.n_targets
            remainder = len(perm) % self.n_targets

            start = 0
            for i, target_label in enumerate(self.trigger_labels):
                extra = 1 if i < remainder else 0
                end = start + n_per_target + extra
                target_perm = perm[start:end]
                start = end

                # Assign target label
                new_targets[target_perm] = target_label

                # Apply trigger based on attack type
                if self.attack_type == 'static':
                    frame_indices = self.frame_assignments[target_label]
                    new_data[target_perm] = self._inject_static_trigger_in_frames(
                        new_data[target_perm], frame_indices, size_width, size_height, width, height)
                    print(f"  -> Target {target_label}: {len(target_perm)} samples, "
                          f"frames {frame_indices}")

                elif self.attack_type == 'latency':
                    delay = self.delay_assignments[target_label]
                    new_data[target_perm] = self._apply_latency_trigger(
                        new_data[target_perm], delay)
                    print(f"  -> Target {target_label}: {len(target_perm)} samples, "
                          f"delay={delay} frames")

                elif self.attack_type == 'rate':
                    rate = self.rate_assignments[target_label]
                    new_data[target_perm] = self._apply_rate_trigger(
                        new_data[target_perm], rate)
                    print(f"  -> Target {target_label}: {len(target_perm)} samples, "
                          f"rate={rate}x")

                elif self.attack_type == 'jitter':
                    jitter_std = self.jitter_assignments[target_label]
                    new_data[target_perm] = self._apply_jitter_trigger(
                        new_data[target_perm], jitter_std)
                    print(f"  -> Target {target_label}: {len(target_perm)} samples, "
                          f"jitter_std={jitter_std}")

                elif self.attack_type == 'jitter_fixed':
                    n_shift = self.nshift_assignments[target_label]
                    new_data[target_perm] = self._apply_jitter_fixed_trigger(
                        new_data[target_perm], n_shift)
                    print(f"  -> Target {target_label}: {len(target_perm)} samples, "
                          f"n_shift={n_shift}")

                for idx in target_perm:
                    poison_target_indices[idx] = target_label

            # Save visualization
            frame = torch.tensor(new_data[perm[0]])
            play_frame(frame, f'backdoor_multi_{self.attack_type}_{mode}.gif')

        print(f'Injecting Over: Bad Imgs: {len(perm)}. '
              f'Clean Imgs: {len(new_data) - len(perm)}. Epsilon: {epsilon}')

        return torch.Tensor(new_data), new_targets, poison_target_indices

    # ========================
    # STATIC TRIGGER
    # ========================
    def _inject_static_trigger_in_frames(self, data, frame_indices, size_width, size_height,
                                          width, height):
        """Inject a static trigger ONLY in the specified frame indices."""
        pos = self.pos
        polarity = self.polarity

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
            x_begin = np.random.randint(0, int(width - size_width))
            x_end = x_begin + size_width
            y_begin = np.random.randint(0, int(height - size_height))
            y_end = y_begin + size_height

        for t in frame_indices:
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

    # ========================
    # LATENCY TRIGGER
    # ========================
    def _apply_latency_trigger(self, data, delay_frames):
        """Apply latency trigger: shift all frames by delay_frames."""
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
        """Apply rate scaling trigger: change temporal playback speed."""
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
        Apply temporal jitter trigger: randomly perturb frame ordering.
        Each target gets a different jitter_std.
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
    # JITTER FIXED TRIGGER
    # ========================
    def _apply_jitter_fixed_trigger(self, data, n_shift):
        """
        Apply controlled frame displacement trigger.
        Each target gets a different n_shift (number of frame-pair swaps).
        """
        N, T, C, H, W = data.shape
        backdoor_data = np.copy(data)

        fixed_indices = np.arange(T)

        step = max(1, T // (n_shift * 2))
        swap_pairs = []

        for k in range(n_shift):
            i = k * step * 2
            j = i + step
            if j < T:
                fixed_indices[i], fixed_indices[j] = fixed_indices[j], fixed_indices[i]
                swap_pairs.append((i, j))

        for n in range(N):
            backdoor_data[n, :, :, :, :] = data[n, fixed_indices, :, :, :]

        print(f"    [jitter_fixed] swaps={swap_pairs}, indices={fixed_indices.tolist()}")
        return backdoor_data


class MultiTargetTestDataset(Dataset):
    """
    Test dataset for a SPECIFIC target in a multi-target attack.
    All samples are poisoned with the trigger for one particular target (epsilon=1).
    """

    def __init__(self, dataset, target_label, target_param, attack_type='static',
                 pos='top-left', time_step=16, trigger_size=0.1, dataname='mnist',
                 polarity=0):
        if type(dataset) == torch.utils.data.Subset:
            path_targets = os.path.join('data/', dataname, f'{time_step}_test_targets.pt')
            path_data = os.path.join('data/', dataname, f'{time_step}_test_data.pt')

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
        self.target_label = target_label
        self.target_param = target_param
        self.attack_type = attack_type
        self.pos = pos
        self.polarity = polarity
        self.trigger_size = trigger_size

        self.data, self.targets = self._poison_all()
        self.channels, self.width, self.height = self.data.shape[2:]

    def _poison_all(self):
        """Poison ALL samples with the trigger for this specific target."""
        print(f"[!] Generating test set for target {self.target_label} "
              f"({self.attack_type}, param={self.target_param})")

        new_data = copy.deepcopy(self.data)
        new_targets = copy.deepcopy(self.targets)

        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)
        if torch.is_tensor(new_data):
            new_data = new_data.numpy()

        new_targets[:] = self.target_label

        if self.attack_type == 'static':
            width, height = new_data.shape[3:]
            size_width = int(self.trigger_size * width)
            size_height = int(self.trigger_size * height)
            if size_width == 0:
                size_width = 1
                size_height = 1
            new_data = self._inject_static_trigger_in_frames(
                new_data, self.target_param, size_width, size_height, width, height)

        elif self.attack_type == 'latency':
            new_data = self._apply_latency_trigger(new_data, self.target_param)

        elif self.attack_type == 'rate':
            new_data = self._apply_rate_trigger(new_data, self.target_param)

        elif self.attack_type == 'jitter':
            new_data = self._apply_jitter_trigger(new_data, self.target_param)

        elif self.attack_type == 'jitter_fixed':
            new_data = self._apply_jitter_fixed_trigger(new_data, self.target_param)

        return torch.Tensor(new_data), new_targets

    # --- Static trigger ---
    def _inject_static_trigger_in_frames(self, data, frame_indices, size_width, size_height,
                                          width, height):
        pos = self.pos
        polarity = self.polarity

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
            x_begin = np.random.randint(0, int(width - size_width))
            x_end = x_begin + size_width
            y_begin = np.random.randint(0, int(height - size_height))
            y_end = y_begin + size_height

        for t in frame_indices:
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

    # --- Latency trigger ---
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

    # --- Rate trigger ---
    def _apply_rate_trigger(self, data, scale_factor):
        N, T, C, H, W = data.shape
        original_indices = np.arange(T)
        new_indices = original_indices * scale_factor
        scaled_indices = np.clip(new_indices, 0, T - 1).astype(int)
        backdoor_data = data[:, scaled_indices, :, :, :]
        return backdoor_data

    # --- Jitter trigger ---
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

    # --- Jitter fixed trigger ---
    def _apply_jitter_fixed_trigger(self, data, n_shift):
        N, T, C, H, W = data.shape
        backdoor_data = np.copy(data)
        fixed_indices = np.arange(T)

        step = max(1, T // (n_shift * 2))
        swap_pairs = []
        for k in range(n_shift):
            i = k * step * 2
            j = i + step
            if j < T:
                fixed_indices[i], fixed_indices[j] = fixed_indices[j], fixed_indices[i]
                swap_pairs.append((i, j))

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
            path_targets = os.path.join('data/', dataname, f'{time_step}_test_targets.pt')
            path_data = os.path.join('data/', dataname, f'{time_step}_test_data.pt')

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


# ================================================================
# DATA LOADER FACTORY
# ================================================================
def create_multi_target_backdoor_data_loader(args):
    """
    Create data loaders for multi-target backdoor attack.
    
    Returns:
        poison_trainloader, clean_testloader, poison_testloaders, target_params
    """
    train_data, test_data = get_dataset(args.dataset, args.T, args.data_dir)

    trigger_labels = args.trigger_labels

    # Build kwargs — pass explicit params if provided, else pass min/max
    extra_kwargs = {}
    if args.type == 'latency':
        if args.delay_params is not None:
            extra_kwargs['delay_params'] = args.delay_params
        else:
            extra_kwargs['delay_min'] = args.delay_min
            extra_kwargs['delay_max'] = args.delay_max

    elif args.type == 'rate':
        if args.rate_params is not None:
            extra_kwargs['rate_params'] = args.rate_params
        else:
            extra_kwargs['rate_min'] = args.rate_min
            extra_kwargs['rate_max'] = args.rate_max

    elif args.type == 'jitter':
        if args.jitter_params is not None:
            extra_kwargs['jitter_params'] = args.jitter_params
        else:
            extra_kwargs['jitter_std_min'] = args.jitter_std_min
            extra_kwargs['jitter_std_max'] = args.jitter_std_max

    elif args.type == 'jitter_fixed':
        if args.nshift_params is not None:
            extra_kwargs['nshift_params'] = args.nshift_params
        else:
            extra_kwargs['n_shift_min'] = args.n_shift_min
            extra_kwargs['n_shift_max'] = args.n_shift_max

    # Create poisoned training set
    train_dataset = MultiTargetPoisonedDataset(
        train_data, trigger_labels=trigger_labels, mode='train',
        epsilon=args.epsilon, pos=args.pos, time_step=args.T,
        trigger_size=args.trigger_size, dataname=args.dataset,
        polarity=args.polarity, attack_type=args.type,
        **extra_kwargs
    )

    # Extract per-target parameters from training dataset
    target_params = train_dataset.target_params

    # Clean test set
    clean_test_dataset = CleanTestDataset(
        test_data, time_step=args.T, dataname=args.dataset
    )

    # Poisoned test set for EACH target
    poison_test_datasets = {}
    for target_label in trigger_labels:
        poison_test_datasets[target_label] = MultiTargetTestDataset(
            test_data, target_label=target_label,
            target_param=target_params[target_label],
            attack_type=args.type,
            pos=args.pos, time_step=args.T,
            trigger_size=args.trigger_size, dataname=args.dataset,
            polarity=args.polarity
        )

    # Data loaders
    poison_trainloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    clean_testloader = DataLoader(
        dataset=clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    poison_testloaders = {}
    for target_label in trigger_labels:
        poison_testloaders[target_label] = DataLoader(
            dataset=poison_test_datasets[target_label],
            batch_size=args.batch_size, shuffle=False, num_workers=1)

    return poison_trainloader, clean_testloader, poison_testloaders, target_params
