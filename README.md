# ClustGDD
The code for paper "Simple yet Effective Graph Distillation via Clustering"

## Brief Introduction
This paper proposes a simple yet effective approach ClustGDD for graph data distillation. ClustGDD achieves superior performance in condensation effectiveness and efficiency over previous GDD solutions on
real datasets through two major contributions: a simple clustering method minimizing the WCSS and a lightweight module augmenting synthetic attributes with class-relevant features. 

## environment
```
numpy                     1.26.0
matplotlib                3.8.2
ogb                       1.3.6 
python                    3.11.5
scikit-learn              1.3.2
torch                     2.0.1                  
torch-cluster             1.6.1+pt20cu118          
torch-geometric           2.6.1                   
torch-scatter             2.1.2+pt20cu118         
torch-sparse              0.6.17+pt20cu118       
torch-spline-conv         1.2.2+pt20cu118
```
PS: deep_robust_data.py and deep_robust_uitls.py are two scource code deeprobust.graph.data and deeprobust.graph.utils from repo deeprobust, you can install it by
```
pip install deeprobust
```
and see the code. 

## Datasets

See the datasets used in https://github.com/ChandlerBang/GCond 

## Runs
```
sh main_induct.sh
sh main_transduct.sh
```

## Reference 
https://github.com/ChandlerBang/GCond 

https://github.com/Amanda-Zheng/SFGC

https://github.com/zclzcl0223/GCSR

https://github.com/liuyang-tian/GDEM


