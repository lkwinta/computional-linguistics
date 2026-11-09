python train_transformer.py --feed-forward mlp --steps 30000 
python train_transformer.py --feed-forward swiglu --use-rope --steps 30000
python train_rnn.py --type rnn --steps 30000
python train_rnn.py --type lstm --steps 30000
