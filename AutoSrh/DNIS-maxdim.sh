#python -u main_nas.py --data_path data/ml-20m/ratings.csv --exp DNIS-CF-dim8 --dataset_type cf --embedding_dim 8 --init_alpha 1.0 --warm_start_epochs 20 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 8 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
#--model_name FM ;
#
#python -u main_nas.py --data_path data/ml-20m/ratings.csv --exp DNIS-CF-dim16 --dataset_type cf --embedding_dim 16 --init_alpha 1.0 --warm_start_epochs 20 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 16 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
#--model_name FM ;
#
#python -u main_nas.py --data_path data/ml-20m/ratings.csv --exp DNIS-CF-dim32 --dataset_type cf --embedding_dim 32 --init_alpha 1.0 --warm_start_epochs 20 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 32 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
#--model_name FM ;
#
#python -u main_nas.py --data_path data/ml-20m/ratings.csv --exp DNIS-CF-dim128 --dataset_type cf --embedding_dim 128 --init_alpha 1.0 --warm_start_epochs 20 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 128 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
#--model_name FM ;
#
#python -u main_nas.py --data_path data/ml-20m/ratings.csv --exp DNIS-CF-dim256 --dataset_type cf --embedding_dim 256 --init_alpha 1.0 --warm_start_epochs 20 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 256 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
#--model_name FM ;

python -u main_nas.py --cuda 0 --data_path data/ml-20m/ratings.csv --exp DNIS-CF-dim512 --dataset_type cf --embedding_dim 512 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 512 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;

python -u main_nas.py --cuda 0 --data_path data/ml-20m/ratings.csv --exp DNIS-CF-dim1024 --dataset_type cf --embedding_dim 1024 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 1024 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;





python -u main_nas.py --cuda 0 --exp DNIS-CTR-dim8 --embedding_dim 8 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 8 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;

python -u main_nas.py --cuda 0 --exp DNIS-CTR-dim16 --embedding_dim 16 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 16 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;

python -u main_nas.py --cuda 0 --exp DNIS-CTR-dim32  --embedding_dim 32 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 32 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;

python -u main_nas.py --cuda 0 --exp DNIS-CTR-dim128 --embedding_dim 128 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 128 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;

python -u main_nas.py --cuda 0 --exp DNIS-CTR-dim256  --embedding_dim 256 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 256 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;

python -u main_nas.py --cuda 0 --exp DNIS-CTR-dim512 --embedding_dim 512 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 512 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;

python -u main_nas.py --cuda 0 --exp DNIS-CTR-dim1024 --embedding_dim 1024 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 1024 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] \
--model_name FM ;