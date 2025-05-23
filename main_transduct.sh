# cora
python train_clustgdd_transduct.py \
  --gpu_id 1 --dataset cora \
  --reduction_rate 1.0 \
  --prop_num 20 \
  --alpha 0.8 --predropout 0.6 \
    --sp_ratio 0.4  --preep 80 \
     --postep 2000 --postprop_num 2 \
      --frcoe 0.01 --predcoe 1.0 --save 1 

python train_clustgdd_transduct.py \
 --gpu_id 1 --dataset cora \
 --reduction_rate 0.5 \
  --prop_num 5 --postprop_num 2 \
   --alpha 0.8 --predropout 0.6 \
    --sp_ratio 0.4  --preep 80 \
     --postep 2000 --postprop_num 2 \
      --frcoe 0.01 --predcoe 1.0 --save 1 

python train_clustgdd_transduct.py \
 --gpu_id 1 --dataset cora \
 --reduction_rate 0.25 \
  --prop_num 5 --postprop_num 2 \
   --alpha 0.8 --predropout 0.7 \
    --sp_ratio 0.06  --preep 80 \
     --postep 2000 --postprop_num 2 \
      --frcoe 0.01 --predcoe 1.0 --save 1 

#citeseer     
python train_clustgdd_transduct.py \
 --gpu_id 1 --dataset citeseer \
 --reduction_rate 0.25 \
  --prop_num 2 --postprop_num 1 \
   --alpha 0.8 --predropout 0.7 \
    --sp_ratio 0.06  --preep 120 \
     --postep 80 --postprop_num 1 \
      --frcoe 0.01 --predcoe 1.0 --save 1 

python train_clustgdd_transduct.py  --gpu_id 1 \
 --dataset citeseer  --reduction_rate 0.5   --prop_num 2 --postprop_num 1  \
   --alpha 0.5 --predropout 0.8 --prewd 0.0002    --sp_ratio 0.21  --preep 100   \
      --postep 200 --hidden 128    \
    --frcoe 0.01 --predcoe 0.05 --save 1  


python train_clustgdd_transduct.py \
 --gpu_id 1  --dataset citeseer \
 --reduction_rate 1.0 \
  --prop_num 2 --postprop_num 1 \
   --alpha 0.5 --predropout 0.7 \
    --sp_ratio 0.2  --preep 200 \
     --postep 200 \
      --frcoe 0.01 --predcoe 0.9 --w1 0.1 --save 1  

# ogbn-arxiv 
python train_clustgdd_transduct.py \
 --gpu_id 1 --dataset ogbn-arxiv \
 --reduction_rate 0.001 \
  --prop_num 20 --postprop_num 10 \
   --alpha 0.92 --predropout 0.6 \
    --sp_ratio 0.1  --preep 1000 \
     --postep 1000 \
      --frcoe 1.4 --predcoe 0.0125 --save 1 


python train_clustgdd_transduct.py \
 --gpu_id 1 --dataset ogbn-arxiv  --reduction_rate 0.01   --prop_num 20 --postprop_num 10  \
  --alpha 0.92 --predropout 0.6     --sp_ratio 0.1  --preep 1000  \
  --postep 1000    \
  --frcoe 1.0  --predcoe 0.025 --save 1 

python train_clustgdd_transduct.py \
                        --gpu_id 0 --dataset ogbn-arxiv \
                        --reduction_rate 0.005 \
                        --prop_num 18 --postprop_num 10 \
                        --alpha 0.91 --predropout 0.6 \
                            --sp_ratio 0.1  --preep 1000 \
                            --postep 1000 \
                            --frcoe 1.9 --predcoe 0.025 --save 1 