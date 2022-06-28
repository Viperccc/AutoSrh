#python -u main_nas.py --exp DNIS-CTR --embedding_dim 64 --init_alpha 1.0 --warm_start_epochs 20 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 64 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [51,126,1763,17057,91265,0] \
#--model_name FM;

python -u main_nas.py --exp DNIS-CTR --embedding_dim 64 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 64 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [51,126,1763,17057,91265,0] \
--model_name Wide_and_Deep;

python -u main_nas.py --exp DNIS-CTR --embedding_dim 64 --init_alpha 1.0 --warm_start_epochs 20 \
--lr_a 0.01 --lr_w 0.001 --num_dim_split 64 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [51,126,1763,17057,91265,0] \
--model_name DeepFM;


#python -u main_nas.py --data_path data/ml-20m/ratings.csv --exp DNIS-CF --dataset_type cf --embedding_dim 64 --init_alpha 1.0 --load_checkpoint 1 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 64 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.2,0.2,0.2,0.3] \
#--model_name MLP ;

#python -u main_nas.py --data_path data/ml-20m/ratings.csv --exp DNIS-CF --dataset_type cf --embedding_dim 128 --init_alpha 1.0 --warm_start_epochs 20 \
#--lr_a 0.01 --lr_w 0.001 --num_dim_split 64 --search_space free --alpha_upper_round 0 --alpha_grad_norm 1 --feature_split [0.1,0.2,0.2,0.2,0.3] \
#--model_name NeuMF ;