
#Mutli-trigger Mutli-Target Attacks



python main_mtmt.py --dataset mnist --T 10 --n_targets 5 --target_types latency latency rate rate jitter_fixed --trigger_labels 0 1 2 3 4 --latency_delays 1 2 --rate_scales 0.5 2.0 --jitter_fixed_shifts 3 --epsilon 0.3 --epochs 1 --data_dir 'data/'


python main_mtmt.py --dataset cifar10 --T 10 --n_targets 5 --target_types latency latency rate rate jitter_fixed --trigger_labels 0 1 2 3 4 --latency_delays 1 2 --rate_scales 0.5 2.0 --jitter_fixed_shifts 3 --epsilon 0.3 --epochs 1 --data_dir 'data/'

python main_mtmt.py --dataset caltech --T 10 --n_targets 5 --target_types latency latency rate rate jitter_fixed --trigger_labels 0 1 2 3 4 --latency_delays 1 2 --rate_scales 0.5 2.0 --jitter_fixed_shifts 3 --epsilon 0.3 --epochs 1 --data_dir '/data/'
