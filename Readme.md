Example of running a test:
``` python main.py --train_dataset all_feat --num_layers 15 --lr 0.001 --batch_size 16384 --pred Cosine --test_mode --model light --step_lr_decay --test_dataset ogbl-collab ```

data.py is the file that processes the data.
model.py is the file that contains the model.
minimum.py is the file that contains a pure linear GCN model with no learnable parameters.
mlp_minimum.py is the file that contains the simplest GCN model with learnable parameters possible.
utils.py is the file that contains the utility functions.