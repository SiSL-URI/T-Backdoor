## Sample Codes for the paper: 'T-Backdoor: Exploiting Temporal Redundancy in Neuromorphic Data for Spike-preserving Backdoor Attacks on SNNs'

## Repository layout

├── models.py                          # Architectures for all datasets (shared)
├── datasets.py                        # Neuromorphic dataset loading (shared)
│
├── main_single.py                     # 1. Single-target entry point
├── single_poisoned_dataset.py         #    Trigger implementations + loaders
├── single_utils.py                    #    Train/eval loop, saving, plots
├── single_target.sh                   #    Paper runs
│
├── main_multi_target.py               # 2. Multi-target entry point
├── multi_target_poisoned_dataset.py   #    Per-target parameter assignment
├── multi_target_utils.py              #    Per-target ASR tracking, saving
├── multi_target.sh                    #    Paper runs
│
├── main_mtmt.py                       # 3. Multi-trigger multi-target entry point
├── mtmt_poisoned_dataset.py           #    Per-target trigger-type config
├── mtmt_utils.py                      #    Per-target ASR tracking, saving
└── mtmt.sh                            #    Paper runs
```

---

## Requirements

- Python 3.11
- CUDA-capable GPU (CPU works but is impractically slow)

```bash
python -m venv venv
source venv/bin/activate

pip install torch torchvision          # CUDA build matching your driver
pip install spikingjelly
pip install numpy scipy matplotlib seaborn tqdm
```

Reference environment used for the reported results:

| Package | Version |
| --- | --- |
| Python | 3.11.10 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| numpy | 2.2.6 |
| scipy | 1.16.0 |
| matplotlib | 3.8.4 |
| seaborn | 0.13.2 |
| spikingjelly | latest from source |

## Dataset setup

Download the raw neuromorphic datasets and place each one in its own subdirectory of
`data/`, named exactly as the `--dataset` value:

```
data/
├── mnist/       # N-MNIST
├── cifar10/     # CIFAR10-DVS
├── caltech/     # N-Caltech101
└── gesture/     # DVS128-Gesture
```

### ⚠️ `data/` must be a literal directory in your working directory

## Usage

### 1. Single-target (`main_single.py`)

One trigger, one target class.

```bash
python main_single.py --dataset cifar10 --type latency --delay_frames 1 \
    --T 10 --epochs 100 --epsilon 0.1 --data_dir 'data/'
```

**Core arguments**

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `gesture` | `mnist`, `cifar10`, `caltech`, `gesture` |
| `--type` | `static` | `static`, `moving`, `smart`, `latency`, `rate`, `jitter`, `jitter_fixed` |
| `--epsilon` | `0.1` | Fraction of the training set to poison |
| `--trigger_label` | `0` | Target class |
| `--T` | `10` | Simulation time-steps |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--loss` | `mse` | `mse` or `cross` |
| `--optim` | `adam` | `adam` or `sgd` |
| `--amp` | off | Enable automatic mixed precision |
| `--seed` | `42` | Random seed |
| `--data_dir` | `data/` | Dataset root |
| `--save_path` | `experiments_single` | Output root |
| `--model_path` | `None` | Start from a pretrained checkpoint |

**Trigger-specific arguments**

| Argument | Default | Applies to |
| --- | --- | --- |
| `--delay_frames` | `2` | `latency` — frames of delay |
| `--scale_factor` | `0.5` | `rate` — `<1` slower, `>1` faster |
| `--n_shift` | `3` | `jitter_fixed` — number of frame pairs swapped |
| `--trigger_size` | `0.1` | `static`/`moving`/`smart` — patch size as a fraction of width |
| `--pos` | `top-left` | `static` — `top-left`, `top-right`, `bottom-left`, `bottom-right`, `middle`, `random` |
| `--polarity` | `3` | `static`/`moving`/`smart` — `0`=black, `1`=dark blue, `2`=green, `3`=light blue |
| `--n_masks` | `2` | `smart` — number of candidate regions |
| `--least` | off | `smart` — target the *least* active region instead of the most |
| `--most_polarity` | off | `smart` — use the most active polarity in the region |

Setting `--epsilon 0.0` trains a clean baseline model.

---

### 2. Multi-target (`main_multi_target.py`)

**One** trigger type, N target classes separated by the trigger's *parameter value*.

```bash
# Latency: delay 1 -> class 0, delay 3 -> class 1, delay 5 -> class 2
python main_multi_target.py --dataset mnist --type latency --n_targets 3 \
    --trigger_labels 0 1 2 --delay_params 1 3 5 \
    --epsilon 0.2 --T 10 --epochs 10 --data_dir 'data/'
```

| Argument | Default | Description |
| --- | --- | --- |
| `--type` | `rate` | `static`, `latency`, `rate`, `jitter`, `jitter_fixed` |
| `--n_targets` | `3` | Number of target classes |
| `--trigger_labels` | `[0..n-1]` | Explicit target labels; length must equal `--n_targets` |
| `--epsilon` | `0.1` | **Total** budget, split evenly across targets |

**Per-target parameters.** Give them explicitly, or let them be auto-assigned by
`np.linspace` over a range. Explicit lists must have length `--n_targets` and override
the range arguments.

| Trigger | Explicit | Auto range | Auto default |
| --- | --- | --- | --- |
| `latency` | `--delay_params 1 3 5` | `--delay_min`, `--delay_max` | `1` … `T//2` |
| `rate` | `--rate_params 0.3 0.6 0.9` | `--rate_min`, `--rate_max` | `0.1` … `6.0` |
| `jitter` | `--jitter_params 0.05 0.15 0.25` | `--jitter_std_min`, `--jitter_std_max` | `0.05` … `0.3` |
| `jitter_fixed` | `--nshift_params 1 3 5` | `--n_shift_min`, `--n_shift_max` | `1` … `max(n_shift_min+1, T//4)` |
| `static` | — | — | `T` split into `n_targets` frame groups |

Auto-assigned integer parameters (`latency`, `jitter_fixed`) are de-duplicated by
incrementing collisions, so no two targets share a value.

For `static`, targets are separated by **which frames carry the patch** — the time axis
distinguishes them rather than the patch appearance.

Per-target ASR is printed every epoch, alongside clean accuracy.

---

### 3. Multi-trigger multi-target (`main_mtmt.py`)

A **different trigger type** per target.

```bash
python main_mtmt.py --dataset mnist --T 10 --n_targets 5 \
    --target_types latency latency rate rate jitter_fixed \
    --trigger_labels 0 1 2 3 4 \
    --latency_delays 1 2 --rate_scales 0.5 2.0 --jitter_fixed_shifts 3 \
    --epsilon 0.3 --epochs 100 --data_dir 'data/'
```

| Argument | Default | Description |
| --- | --- | --- |
| `--n_targets` | `3` | Number of targets; `3` and `5` have built-in presets |
| `--target_types` | `None` | Trigger type per target — `static`, `latency`, `rate`, `jitter`, `jitter_fixed`. Length must equal `--n_targets`. Overrides the presets |
| `--trigger_labels` | `[0..n-1]` | Target labels |
| `--epsilon` | `0.2` | **Total** budget, split evenly across targets |

**Per-type parameter lists.** Each list has one entry per target *of that type* — with
`--target_types latency latency rate rate jitter_fixed` you pass two delays, two rates
and one shift:

| Argument | Applies to |
| --- | --- |
| `--latency_delays 1 2` | each `latency` target |
| `--rate_scales 0.5 2.0` | each `rate` target |
| `--jitter_fixed_shifts 3` | each `jitter_fixed` target |
| `--jitter_stds 0.1 0.3` | each `jitter` target |
| `--static_positions top-left bottom-right` | each `static` target |
| `--static_polarities 3 1` | each `static` target |

Omitting a list auto-assigns from the matching `*_min` / `*_max` range.

**Built-in presets** (used when `--target_types` is not given):

- `--n_targets 3` — `static` (frames `0..T/3`), `latency` (`--delay`), `rate` (`--scale_factor`)
- `--n_targets 5` — two `static` (different frame groups and positions), two `latency`
  (`--delay`, `--delay_2`), one `rate`

Any other `--n_targets` value **requires** `--target_types`.

---

## Outputs

Every run writes a row to a shared CSV plus a self-contained experiment directory.

| Setting | Summary CSV | Experiment directory |
| --- | --- | --- |
| Single | `<save_path>/results.csv` | `<dataset>_<type>_<eps>_<size>_<param>_<seed>/` |
| Multi-target | `<save_path>/multi_target_results.csv` | `multi_<type>_<dataset>_<n>targets_<labels>_eps<eps>_<params>_seed<seed>/` |
| MTMT | `<save_path>/mtmt_results.csv` | `mtmt_<dataset>_<n>t_<params>_eps<eps>_seed<seed>/` |

Trigger parameters are encoded into the directory name (`df1` = delay 1, `sf0.1` = rate
0.1, `ns3` = 3 swaps), so runs that differ only in a trigger parameter do not overwrite
each other.



