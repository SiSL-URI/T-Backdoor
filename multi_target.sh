

#Rate Trigger

#------------------------------------------------------------------MNIST-----------------------------------------------------------------------#

python main_multi_target.py --dataset mnist --type rate --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --rate_min 0.1 --rate_max 4.0 --epochs 10 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset mnist --type rate --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --rate_min 0.1 --rate_max 4.0 --epochs 10 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset mnist --type rate --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --rate_min 0.1 --rate_max 6.0 --epochs 10 --T 10 --data_dir 'data/'



#------------------------------------------------------------------------CIFAR-10----------------------------------------------------------------------------#

python main_multi_target.py --dataset cifar10 --type rate --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --rate_min 0.1 --rate_max 4.0 --epochs 100 --T 20 --data_dir 'data/'
python main_multi_target.py --dataset cifar10 --type rate --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --rate_min 0.1 --rate_max 4.0 --epochs 100 --T 20 --data_dir 'data/'
python main_multi_target.py --dataset cifar10 --type rate --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --rate_min 0.1 --rate_max 6.0 --epochs 100 --T 20 --data_dir 'data/'


#------------------------------------------------------------------------Caltech----------------------------------------------------------------------------#

python main_multi_target.py --dataset caltech --type rate --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --rate_min 0.1 --rate_max 4.0 --epochs 100 --T 5 --data_dir 'data/'
python main_multi_target.py --dataset caltech --type rate --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --rate_min 0.1 --rate_max 4.0 --epochs 100 --T 5 --data_dir 'data/'
python main_multi_target.py --dataset caltech --type rate --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --rate_min 0.1 --rate_max 6.0 --epochs 100 --T 5 --data_dir 'data/'



#Latency Trigger


#------------------------------------------------------------------MNIST-----------------------------------------------------------------------#

python main_multi_target.py --dataset mnist --type latency --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --delay_params 1 3 5  --epochs 10 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset mnist --type latency --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --delay_params 1 2 4 6   --epochs 10 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset mnist --type latency --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --delay_params 1 2 3 4 5 --epochs 10 --T 10 --data_dir 'data/'




#------------------------------------------------------------------------CIFAR-10----------------------------------------------------------------------------#

python main_multi_target.py --dataset cifar10 --type latency --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --delay_params 1 3 5  --epochs 100 --T 20 --data_dir 'data/'
python main_multi_target.py --dataset cifar10 --type latency --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --delay_params 1 2 4 6   --epochs 100 --T 20 --data_dir 'data/'
python main_multi_target.py --dataset cifar10 --type latency --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --delay_params 1 2 3 4 5 --epochs 100 --T 20 --data_dir 'data/'

#------------------------------------------------------------------------Caltech----------------------------------------------------------------------------#

python main_multi_target.py --dataset caltech --type latency --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --delay_params 1 3 5  --epochs 100 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset caltech --type latency --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --delay_params 1 2 4 6   --epochs 100 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset caltech --type latency --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --delay_params 1 2 3 4 5 --epochs 100 --T 10 --data_dir 'data/'


#Jitter Trigger

#------------------------------------------------------------------MNIST-----------------------------------------------------------------------#

python main_multi_target.py --dataset mnist --type jitter_fixed --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --nshift_params 1 2 3 --epochs 10 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset mnist --type jitter_fixed --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --nshift_params 1 2 3 4  --epochs 10 --T 10 --data_dir 'data/'
python main_multi_target.py --dataset mnist --type jitter_fixed --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --nshift_params 1 2 3 4 5 --epochs 10 --T 10 --data_dir 'data/'




#------------------------------------------------------------------CIFAR-10-----------------------------------------------------------------------#

python main_multi_target.py --dataset cifar10 --type jitter_fixed --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --nshift_params 1 2 3 --epochs 100 --T 20 --data_dir 'data/'
python main_multi_target.py --dataset cifar10 --type jitter_fixed --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --nshift_params 1 2 3 4  --epochs 100 --T 20 --data_dir 'data/'
python main_multi_target.py --dataset cifar10 --type jitter_fixed --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --nshift_params 1 2 3 4 5 --epochs 100 --T 20 --data_dir 'data/'

#------------------------------------------------------------------CALTECH-----------------------------------------------------------------------#


python main_multi_target.py --dataset caltech --type jitter_fixed --n_targets 3 --trigger_labels 0 1 2 --epsilon 0.2 --nshift_params 1 2 3 --epochs 100 --T 5 --data_dir 'data/'
python main_multi_target.py --dataset caltech --type jitter_fixed --n_targets 4 --trigger_labels 0 1 2 3 --epsilon 0.2 --nshift_params 1 2 3 4  --epochs 100 --T 5 --data_dir 'data/'
python main_multi_target.py --dataset caltech --type jitter_fixed --n_targets 5 --trigger_labels 0 1 2 3 4 --epsilon 0.2 --nshift_params 1 2 3 4 5 --epochs 100 --T 10 --data_dir 'data/'







































