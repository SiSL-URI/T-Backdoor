import torch.nn.functional as F
import os
from spikingjelly.datasets import play_frame
from datasets import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import copy
from torchvision import transforms


class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label=0, mode='train', epsilon=0.1, pos='top-left', attack_type='static', time_step=16,
                 trigger_size=0.1, dataname='mnist', polarity=0, n_masks=2, least=False, most_polarity=False,
                 delay_frames=2, scale_factor=0.5, n_shift=3):

        # Handle special case for CIFAR10 and Caltech101
        if type(dataset) == torch.utils.data.Subset:
            path_targets = os.path.join(
                'data/', dataname, f'{time_step}_{mode}_targets.pt')
            path_data = os.path.join(
                'data/', dataname, f'{time_step}_{mode}_data.pt')

            if os.path.exists(path_targets) and os.path.exists(path_data):
                targets = torch.load(path_targets)
                data = torch.load(path_data)
            else:

                targets = torch.Tensor(dataset.dataset.targets)[
                    dataset.indices]
                if dataset.dataset[0][0].shape[-1] != dataset.dataset[0][0].shape[-2]:
                    crop = transforms.CenterCrop(
                        min(dataset.dataset[0][0].shape[-1], dataset.dataset[0][0].shape[-2]))
                    data = np.array([crop(torch.Tensor(i[0])).numpy()
                                    for i in dataset.dataset])
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
            # We need the images loaded instead of the paths
            self.data = np.array([np.array(x[0])
                                 for x in dataset])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.time_step = time_step
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform
        self.trigger_label = trigger_label
        self.least = least
        self.most_polarity = most_polarity
        self.pos = pos
        self.polarity = polarity
        self.n_masks = n_masks

        # Store temporal trigger parameters
        self.delay_frames = delay_frames
        self.scale_factor = scale_factor
        self.n_shift = n_shift

        self.data, self.targets = self.add_trigger(
            trigger_label, epsilon, mode, attack_type, trigger_size
        )

        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        # Retrieves the requested image applying the transformations and the label is one-hot encoded
        img = self.data[item]

        targets = self.targets[item]
        if self.transform:
            img = self.transform(img)

        return img, F.one_hot(targets.long(), self.class_num).float()

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[2:]

    def add_trigger(self, trigger_label, epsilon, mode, type, trigger_size):

        print("[!] Generating " + mode + " Bad Imgs")

        new_data = copy.deepcopy(self.data)
        new_targets = copy.deepcopy(self.targets)

        # Ensure that targets are tensors
        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)

        # Choose a random subset of samples to be poisoned
        perm = np.random.permutation(len(new_data))[
            0: int(len(new_data) * epsilon)]

        # Ensure that new_data is a np.array
        if torch.is_tensor(new_data):
            new_data = new_data.numpy()

        width, height = new_data.shape[3:]

        # Swap every samples to the target class
        new_targets[perm] = trigger_label

        size_width = int(trigger_size * width)
        size_height = int(trigger_size * height)

        if epsilon != 0.0:
            if size_width == 0:
                size_width = 1
                size_height = 1

        if len(perm) != 0:
            if type == 'static':
                new_data[perm] = self.create_static_trigger(
                    new_data[perm], size_width, size_height, width, height)
            elif type == 'moving':
                new_data[perm] = self.create_moving_trigger(
                    new_data[perm], size_width, size_height, height, width)
            elif type == 'smart':
                new_data[perm] = self.create_smart_trigger(
                    new_data[perm], size_height, height, width, new_data)
                
            #TEMPORAL TRIGGERS
            elif type == 'latency':
                new_data[perm] = self.create_latency_trigger(new_data[perm], self.delay_frames)
                
            elif type == 'rate':
                new_data[perm] = self.create_rate_scaling_trigger(new_data[perm], self.scale_factor)
                
            elif type == 'jitter':
                jitter_std = 0.2  # Default std for jitter
                new_data[perm] = self.create_jitter_trigger(new_data[perm], jitter_std)
            
            elif type == 'jitter_fixed':
                new_data[perm] = self.create_jitter_fixed_trigger(new_data[perm], self.n_shift)
                
            elif type == 'burst':
                burst_frames = 2  # Frames to duplicate for burst
                new_data[perm] = self.create_burst_trigger(new_data[perm], burst_frames)
            else:
                raise Exception('Invalid Trigger Type')

            frame = torch.tensor(new_data[perm][0])
            play_frame(frame, f'backdoor_{type}.gif')

        print(
            f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(new_data)-len(perm)}. Epsilon: {epsilon}')

        return torch.Tensor(new_data), new_targets
    
    # NEW TEMPORAL TRIGGER METHODS
    def create_latency_trigger(self, data, delay_frames):
        """
        Temporal Trigger 1: Add latency by shifting frames
        
        Args:
            data: Shape (N, T, C, H, W)
            delay_frames: Number of frames to delay
        
        Returns:
            Modified data with latency trigger
        """
        N, T, C, H, W = data.shape
        backdoor_data = np.zeros_like(data)
        
        # Shift all frames by delay_frames
        if delay_frames < T:
            # Copy delayed frames
            backdoor_data[:, delay_frames:, :, :, :] = data[:, :-delay_frames, :, :, :]
            # Fill first delay_frames with zeros (or copy first frame)
            for i in range(delay_frames):
                backdoor_data[:, i, :, :, :] = data[:, 0, :, :, :]  # Repeat first frame
        else:
            # If delay is too large, just repeat first frame
            for i in range(T):
                backdoor_data[:, i, :, :, :] = data[:, 0, :, :, :]
        
        print(f"Applied latency trigger: {delay_frames} frame delay")
        return backdoor_data
    
    def create_rate_scaling_trigger(self, data, scale_factor):
        """
        Temporal Trigger 2: Scale temporal rate (speed up/slow down)
        
        Args:
            data: Shape (N, T, C, H, W)
            scale_factor: <1.0 for faster, >1.0 for slower
        
        Returns:
            Modified data with rate scaling
        """
        N, T, C, H, W = data.shape
        backdoor_data = np.zeros_like(data)
        
        # Create new temporal indices
        original_indices = np.arange(T)
        new_indices = original_indices * scale_factor
        
        # Interpolate to new temporal positions
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        # Simple nearest neighbor temporal interpolation
                        scaled_indices = np.clip(new_indices, 0, T-1).astype(int)
                        backdoor_data[n, :, c, h, w] = data[n, scaled_indices, c, h, w]
        
        print(f"Applied rate scaling trigger: {scale_factor}x speed")
        return backdoor_data
    
    def create_jitter_trigger(self, data, jitter_std):
        """
        Temporal Trigger 3: Add temporal jitter/noise
        
        Args:
            data: Shape (N, T, C, H, W)
            jitter_std: Standard deviation for jitter (relative to T)
        
        Returns:
            Modified data with temporal jitter
        """
        N, T, C, H, W = data.shape
        backdoor_data = np.zeros_like(data)
        
        # Generate jitter for each sample
        for n in range(N):
            # Create jittered temporal mapping
            jitter = np.random.normal(0, jitter_std * T, T)
            jittered_indices = np.arange(T) + jitter
            jittered_indices = np.clip(jittered_indices, 0, T-1)
            
            # Sort to maintain causality
            sorted_indices = np.argsort(jittered_indices)
            
            # Apply jittered mapping
            backdoor_data[n, :, :, :, :] = data[n, sorted_indices, :, :, :]
        
        print(f"Applied temporal jitter trigger: std={jitter_std}")
        return backdoor_data
    
    def create_jitter_fixed_trigger(self, data, n_shift=2):
        """
        Temporal Trigger 3: Controlled frame displacement
        
        Args:
            data: Shape (N, T, C, H, W)
            n_shift: Number of frame pairs to swap (controls stealthiness)
                     - 1: swap 1 pair (very stealthy)
                     - 2: swap 2 pairs
                     - T//2: swap all pairs (maximum disruption)
                     Can also pass a list of tuples for exact control:
                       e.g., [(0,2), (5,7)] swaps frame 0↔2 and 5↔7
        
        Returns:
            Modified data with controlled frame displacement
        """
        N, T, C, H, W = data.shape
        backdoor_data = np.copy(data)
        
        fixed_indices = np.arange(T)
        
        if isinstance(n_shift, list):
            # Direct control: user specifies exact swap pairs
            # e.g., [(0, 2), (5, 7)] 
            swap_pairs = n_shift
            for i, j in swap_pairs:
                assert i < T and j < T, f"Index out of range: ({i},{j}), T={T}"
                fixed_indices[i], fixed_indices[j] = fixed_indices[j], fixed_indices[i]
        
        elif isinstance(n_shift, int):
            # Auto-generate n_shift swap pairs, evenly spaced
            available = list(range(T))
            swap_pairs = []
            step = max(1, T // (n_shift * 2))  # Space swaps evenly
            
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
        
        print(f"Applied jitter trigger: swaps={swap_pairs}, indices={fixed_indices}")
        return backdoor_data

    def create_static_trigger(self, data, size_width, size_height, width, height):
        pos = self.pos
        polarity = self.polarity

        if pos == 'top-left':
            x_begin = 0
            x_end = size_width
            y_begin = 0
            y_end = size_height

        elif pos == 'top-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = 0
            y_end = size_height

        elif pos == 'bottom-left':
            x_begin = 0
            x_end = size_width
            y_begin = int(height - size_height)
            y_end = height

        elif pos == 'bottom-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = int(height - size_height)
            y_end = height

        elif pos == 'middle':
            x_begin = int((width - size_width) / 2)
            x_end = int((width + size_width) / 2)
            y_begin = int((height - size_height) / 2)
            y_end = int((height + size_height) / 2)

        elif pos == 'random':
            x_begin = np.random.randint(0, width - size_width)
            x_end = x_begin + size_width
            y_begin = np.random.randint(0, height - size_height)
            y_end = y_begin + size_height

        # [0, 0] Black
        if polarity == 0:
            data[:, :, :, y_begin:y_end, x_begin:x_end] = 0
        # [0, 1] Dark blue
        elif polarity == 1:
            data[:, :, 0, y_begin:y_end, x_begin:x_end] = 0
            data[:, :, 1, y_begin:y_end, x_begin:x_end] = 1
        # [1, 0] Green
        elif polarity == 2:
            data[:, :, 0, y_begin:y_end, x_begin:x_end] = 1
            data[:, :, 1, y_begin:y_end, x_begin:x_end] = 0
        # [1, 1] Light blue
        else:
            data[:, :, :, y_begin:y_end, x_begin:x_end] = 1

        return data

    def create_moving_trigger(self, data, size_width, size_height, height, width):
        pos = self.pos
        polarity = self.polarity

        for t in range(data.shape[1]):
            # Select a random position for the trigger
            if pos == 'random':
                x_begin = np.random.randint(0, width - size_width)
                y_begin = np.random.randint(0, height - size_height)
            else:
                x_begin = np.random.randint(0, width - size_width)
                y_begin = np.random.randint(0, height - size_height)

            x_end = x_begin + size_width
            y_end = y_begin + size_height

            # [0, 0] Black
            if polarity == 0:
                data[:, t, :, y_begin:y_end, x_begin:x_end] = 0
            # [0, 1] Dark blue
            elif polarity == 1:
                data[:, t, 0, y_begin:y_end, x_begin:x_end] = 0
                data[:, t, 1, y_begin:y_end, x_begin:x_end] = 1
            # [1, 0] Green
            elif polarity == 2:
                data[:, t, 0, y_begin:y_end, x_begin:x_end] = 1
                data[:, t, 1, y_begin:y_end, x_begin:x_end] = 0
            # [1, 1] Light blue
            else:
                data[:, t, :, y_begin:y_end, x_begin:x_end] = 1

        return data

    def create_smart_trigger(self, data, t_size, height, width, original_data):

        n_masks = self.n_masks
        least = self.least
        most_polarity = self.most_polarity

        masks = get_masks(height, width, n_masks, n_masks)

        mask = get_most_active_mask(masks, original_data, least)

        safe_h0 = mask[0]
        safe_h1 = mask[1] - t_size
        safe_w0 = mask[2]
        safe_w1 = mask[3] - t_size

        pol_values = np.zeros((masks.shape[0], 4),  dtype=int)
        max_index = np.argmax(
            np.sum(pol_values[:, 1:], axis=1))

        if most_polarity:
            polarity = np.argmax(pol_values[max_index])
        else:
            polarity = np.argmin(pol_values[max_index])

        # For each frame t, we will create a trigger in a random position (close the one before t-1) in between the safe boundaries
        # The trigger will be a square of size t_size

        # Get a random position for the trigger in the mask
        h0 = np.random.randint(safe_h0, safe_h1)
        w0 = np.random.randint(safe_w0, safe_w1)

        r = 2  # The closeness of the trigger to the previous one, in pixels
        for t in range(data.shape[1]):
            # Left or right
            while True:
                if np.random.randint(2) == 0:
                    w0 += r
                else:
                    w0 -= r
                if w0 >= safe_w0 and w0 <= safe_w1:
                    break

            # Up or down
            while True:
                if np.random.randint(2) == 0:
                    h0 += r
                else:
                    h0 -= r
                if h0 >= safe_h0 and h0 <= safe_h1:
                    break

            h1 = h0 + t_size
            w1 = w0 + t_size

            # [0, 0] Black
            if polarity == 0:
                data[:, t, :, h0:h1, w0:w1] = 0
            # [0, 1] Dark blue
            elif polarity == 1:
                data[:, t, 0, h0:h1, w0:w1] = 0
                data[:, t, 1, h0:h1, w0:w1] = 1
            # [1, 0] Green
            elif polarity == 2:
                data[:, t, 0, h0:h1, w0:w1] = 1
                data[:, t, 1, h0:h1, w0:w1] = 0
            # [1, 1] Light blue
            else:
                data[:, t, :, h0:h1, w0:w1] = 1

        return data


def create_backdoor_data_loader(args):

    # Get the dataset
    train_data, test_data = get_dataset(args.dataset, args.T, args.data_dir)

    train_data = PoisonedDataset(train_data, args.trigger_label, mode='train', epsilon=args.epsilon,
                                 pos=args.pos, attack_type=args.type, time_step=args.T,
                                 trigger_size=args.trigger_size, dataname=args.dataset,
                                 polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity,
                                 delay_frames=args.delay_frames, scale_factor=args.scale_factor, n_shift=args.n_shift)

    test_data_ori = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=0,
                                    pos=args.pos, attack_type=args.type, time_step=args.T,
                                    trigger_size=args.trigger_size, dataname=args.dataset,
                                    polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity,
                                    delay_frames=args.delay_frames, scale_factor=args.scale_factor, n_shift=args.n_shift)

    test_data_tri = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=1,
                                    pos=args.pos, attack_type=args.type, time_step=args.T,
                                    trigger_size=args.trigger_size, dataname=args.dataset,
                                    polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity,
                                    delay_frames=args.delay_frames, scale_factor=args.scale_factor, n_shift=args.n_shift)

    frame, label = test_data_tri[0]
    play_frame(frame, 'backdoor.gif')

    train_data_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_data_ori_loader = DataLoader(
        dataset=test_data_ori, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_data_tri_loader = DataLoader(
        dataset=test_data_tri, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader


def create_backdoor_data_loader_adaptive(args):

    # Get the dataset
    train_data, test_data = get_dataset(args.dataset, args.T, args.data_dir)

    clean_trainset = PoisonedDataset(train_data, args.trigger_label, mode='train', epsilon=0,
                                     pos=args.pos, attack_type=args.type, time_step=args.T,
                                     trigger_size=args.trigger_size, dataname=args.dataset,
                                     polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity,
                                     delay_frames=args.delay_frames, scale_factor=args.scale_factor, n_shift=args.n_shift)

    bk_trainset = PoisonedDataset(train_data, args.trigger_label, mode='train', epsilon=1,
                                  pos=args.pos, attack_type=args.type, time_step=args.T,
                                  trigger_size=args.trigger_size, dataname=args.dataset,
                                  polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity,
                                  delay_frames=args.delay_frames, scale_factor=args.scale_factor, n_shift=args.n_shift)

    test_data_ori = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=0,
                                    pos=args.pos, attack_type=args.type, time_step=args.T,
                                    trigger_size=args.trigger_size, dataname=args.dataset,
                                    polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity,
                                    delay_frames=args.delay_frames, scale_factor=args.scale_factor, n_shift=args.n_shift)

    test_data_tri = PoisonedDataset(test_data, args.trigger_label, mode='test', epsilon=1,
                                    pos=args.pos, attack_type=args.type, time_step=args.T,
                                    trigger_size=args.trigger_size, dataname=args.dataset,
                                    polarity=args.polarity, n_masks=args.n_masks, least=args.least, most_polarity=args.most_polarity,
                                    delay_frames=args.delay_frames, scale_factor=args.scale_factor, n_shift=args.n_shift)

    frame, label = test_data_tri[0]
    play_frame(frame, 'backdoor.gif')

    clean_trainloader = DataLoader(
        dataset=clean_trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    bk_trainloader = DataLoader(
        dataset=bk_trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_data_ori_loader = DataLoader(
        dataset=test_data_ori, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_data_tri_loader = DataLoader(
        dataset=test_data_tri, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return clean_trainloader, bk_trainloader, test_data_ori_loader, test_data_tri_loader


def get_masks(H, W, n, n_masks):

    masks = np.zeros((n_masks, 4), dtype=int)

    # Get the size of each mask
    size_h = H/(n+1)
    size_w = W/(n+1)
    x = 0
    y = 0
    for mask in masks:
        mask[0] = y
        mask[1] = y + size_h
        mask[2] = x
        mask[3] = x + size_w
        x += size_w
        if x >= W:
            x = 0
            y += size_h

    return masks


def get_most_active_mask(masks, data, least):
    # data can be a single image or a batch of images
    pol_values = np.zeros((masks.shape[0], 4),  dtype=int)
    max_values = np.zeros((masks.shape[0]), dtype=int)

    # Get the maximum value of each mask
    # Since the image is neuromorphic it shape is [batch, T, ...]

    # If data is a single image, we need to add a dimension for the batch
    # and another for the time
    if len(data.shape) == 3:
        data = data[np.newaxis, np.newaxis, :, :]

    for idx, mask in enumerate(masks):

        # [0,0] case
        p_0 = np.sum(
            data[:, :, :, mask[0]:mask[1], mask[2]:mask[3]] == 0
        )

        # [0,1] case
        p_1 = np.sum(np.logical_and(
            data[:, :, 0, mask[0]:mask[1], mask[2]:mask[3]] == 0,
            data[:, :, 1, mask[0]:mask[1], mask[2]:mask[3]] == 1
        ))

        # [1,0] case
        p_2 = np.sum(np.logical_and(
            data[:, :, 0, mask[0]:mask[1], mask[2]:mask[3]] == 1,
            data[:, :, 1, mask[0]:mask[1], mask[2]:mask[3]] == 0
        ))

        # [1,1] case
        p_3 = np.sum(
            data[:, :, :, mask[0]:mask[1], mask[2]:mask[3]] == 1
        )

        pol_values[idx, 0] = p_0
        pol_values[idx, 1] = p_1
        pol_values[idx, 2] = p_2
        pol_values[idx, 3] = p_3

        # Assuming that p_0 black, i.e., no movement
        # Here we are looking for the more "active" mask
        max_values[idx] = p_1 + p_2 + p_3

    if least:
        max_index = np.argmin(max_values)

    else:
        max_index = np.argmax(max_values)

    # Now we need to create the moving trigger in the mask with the highest value
    # Take into account the trigger size has to be inside the mask
    return masks[max_index]
