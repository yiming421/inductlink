data.py is the file that processes the data.  
model.py is the file that contains the model.  
minimum.py is the file that contains a pure linear GCN model with no learnable parameters.  
mlp_minimum.py is the file that contains the simplest GCN model with learnable parameters possible.  
noisy.py is the file that contains the model with noise adaptors.  
pca.py is the file that contains the model with PCA-unified feature.  
pfn.py is the file that contains the PFN model.  
utils.py is the file that contains the utility functions.  

```python pfn.py --train_dataset CiteSeer,PubMed,CS,Physics,Computers,Photo --test_dataset Cora --lr 0.0001 --epochs 500 --maskinput True --norm True --num_layers 4 --mlp_layers 5 --transformer_layers 10 --nhead 4 --context_num 32 --padding mlp``` command for pca+pfn

```python pca.py --hidden 200 --maskinput True --norm True --predictor Transformer --num_neg 1 --pma True --transformer_layers 1 --pma True --transformer_hidden 16 --epochs 500 --num_heads 1 --train_dataset CiteSeer,PubMed,CS,Physics,Computers,Photo --test_dataset Cora``` command for pca+setTransformer

```python pca.py --hidden 256 --maskinput True --norm True --predictor Transformer --num_neg 3 --epochs 500 --mlp_layers 5 --mlp_res True --train_dataset CiteSeer,PubMed,CS,Physics,Computers,Photo --test_dataset Cora``` command for pca+MLP

```python noise.py --hidden 256 --maskinput True --norm True --predictor Transformer --num_neg 3 --epochs 500 --mlp_layers 5 --mlp_res True --train_dataset CiteSeer,PubMed,CS,Physics,Computers,Photo --test_dataset Cora``` command for noise+MLP

```python mlp_minimum.py --dataset Cora  --maskinput True --norm True``` command for transductive MLP

```python minimum.py --dataset Cora``` command for PureGCN
