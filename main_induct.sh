python train_clustgdd_induct.py --gpu_id 0  --dataset flickr \
  --reduction_rate 0.01    --prop_num 2 --postprop_num 2 \
   --alpha 0.8 --predropout 0.6     --sp_ratio 0.5  --preep 2000   \
     --postep 2000    --frcoe 0.2 --predcoe 0.8 --w1 0.8 --save 1

python train_clustgdd_induct.py --gpu_id 2  --dataset flickr \
  --reduction_rate 0.005    --prop_num 2 --postprop_num 1\
   --alpha 0.8 --predropout 0.6     --sp_ratio 0.5  --preep 2000  \
      --postep 2000   --frcoe 0.2 --predcoe 0.8 --w1 1.0 --save 1  

python train_clustgdd_induct.py --gpu_id 0  --dataset flickr \
  --reduction_rate 0.001    --prop_num 2 --postprop_num 2 \
   --alpha 0.8 --predropout 0.6     --sp_ratio 0.5  --preep 2000  \
      --postep 2000    --frcoe 0.2 --predcoe 0.8 --w1 0.8 --save 1 

python train_clustgdd_induct.py --gpu_id 1  --dataset reddit \
 --reduction_rate 0.005  \
   --prop_num 20 --postprop_num 10    --alpha 0.95 \
    --predropout 0.6     --sp_ratio 0.1 \
     --preep 1000      --postep 1000   \
         --frcoe 0.8 --predcoe 0.05 --w1 0.01 

python train_clustgdd_induct.py --gpu_id 0  --dataset reddit \
  --reduction_rate 0.001     --prop_num 10 --postprop_num 7  \
    --alpha 0.92 --predropout 0.6   \ 
       --sp_ratio 0.1  --preep 800      --postep 600     \
              --hidden 512     --frcoe 3.3  --predcoe 0.024 --w1 0.015 --save 1


python train_clustgdd_induct.py --gpu_id 3 \
                        --dataset reddit \
                        --reduction_rate 0.0005  \
                        --prop_num 10  --postprop_num 7    --alpha 0.92 --predropout 0.6   \
                        --sp_ratio 0.1  --preep 800      --postep 600      \
                        --hidden 512     --frcoe 3.5 \
                        --predcoe 0.038  --w1 0.015 --save 1