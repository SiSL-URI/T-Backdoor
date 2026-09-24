
#n-mnist

python main_single.py --type 'rate' --dataset 'mnist' --T 10 --epoch 10 --epsilon 0.1 --data_dir 'data/'
python main_single.py --type 'latency' --dataset 'mnist' --T 10 --epoch 10 --epsilon 0.1 --data_dir 'data/'
python main_single.py --type 'jitter_fixed' --dataset 'mnist' --T 10 --epoch 10 --epsilon 0.1 --data_dir 'data/'


#cifar10-dvs

python main_single.py --type 'rate' --dataset 'cifar10' --T 10 --epoch 100 --epsilon 0.1 --data_dir 'data/'
python main_single.py --type 'latency' --dataset 'cifar10' --T 10 --epoch 100 --epsilon 0.1 --data_dir 'data/'
python main_single.py --type 'jitter_fixed' --dataset 'cifar10' --T 10 --epoch 100 --epsilon 0.1 --data_dir 'data/'

#n-caltech

python main_single.py --type 'rate' --dataset 'caltech' --T 10 --epoch 100 --epsilon 0.1 --data_dir 'data/'
python main_single.py --type 'latency' --dataset 'caltech' --T 10 --epoch 100 --epsilon 0.1 --data_dir 'data/'
python main_single.py --type 'jitter_fixed' --dataset 'caltech' --T 10 --epoch 100 --epsilon 0.1 --data_dir 'data/'
