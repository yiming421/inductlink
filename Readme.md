Example of running a test:
``` python main.py --train_dataset all_feat --num_layers 15 --lr 0.001 --batch_size 16384 --pred Cosine --test_mode --model light --step_lr_decay --test_dataset ogbl-collab ```

data.py is the file that processes the data.
model.py is the file that contains the model.
main.py is the file that runs the model.
routine.py is the file that contains the training and testing routines.
utils.py is the file that contains the utility functions.

Explanation of the arguments:
--train_dataset: the dataset used for training. all_feat uses multiple datasets with feature for training. all_no_feat uses multiple datasets without feature for training.
--test_dataset: the dataset used for testing.
--num_layers: the number of GNN layers.
--alpha: the decaying factor in lightGCN model.
--test_epochs: the number of epochs for testing, if we need additional epochs for testing.
--pred: the prediction function. possible choices can be found in the line 198 of main.py.
--mlp_layers: the number of layers in the Hadamard prediction function, if we choose MLP as the prediction function.
--test_mode: include this argument if we want to run inductive test.
--model: the model used for training. no_feat uses the model without feature, feat use PCA to reduce the dimension of the feature to a unified dimension, light uses lightGCN model.
--res: whether to use residual connection in the MLP prediction function.
--residual: the residual rate in GNN layers.
--relu: whether to use ReLU activation function between GNN layers.
--linear: whether to use an additional linear layer between GNN layers.
--lda: whether to process the initial feature with LDA.
--pca: whether to process the initial feature with PCA.
--learnable_emb: whether to use learnable embeddings in the training process.
--conv: decide whether we should use GCN or SAGE convolution.
--initial_emb: the initial embeddings used in the training process. eigen uses the eigenvector of the adjacency matrix, pca uses the PCA projection of the adjacency matrix, ones uses the matrix with all elements being 1, noise uses the matrix with orthogonal noise, node2vec uses the node2vec embeddings.
--loss_fn: the loss function used in the training process. bce or auc.
--rand_noise: whether to use a different random noise in each epoch.
--agg_type: the aggregation type in the attention mechanism in lightGCN model. add_agg means directly learn the aggregation weights, mlp_agg uses GraphAny aggregation mechanism.
