cd /fangwenxuan/RL_USP_new

# Norm
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 32 --batch_size_ppo 256 --beta 0.01 --config_actor_deep 4 --config_actor_dim 512 --config_actor_mlp_dim 1024 --config_critic_deep 4 --config_critic_dim 512 --config_critic_mlp_dim 1024

# S
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 32 --batch_size_ppo 256 --beta 0.01 --config_actor_deep 2 --config_actor_dim 512 --config_actor_mlp_dim 1024 --config_critic_deep 2 --config_critic_dim 512 --config_critic_mlp_dim 1024

# XS
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 32 --batch_size_ppo 256 --beta 0.01 --config_actor_deep 2 --config_actor_dim 256 --config_actor_mlp_dim 512 --config_critic_deep 2 --config_critic_dim 256 --config_critic_mlp_dim 512

# XXS
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 32 --batch_size_ppo 256 --beta 0.01 --config_actor_deep 2 --config_actor_dim 128 --config_actor_mlp_dim 256 --config_critic_deep 2 --config_critic_dim 128 --config_critic_mlp_dim 256

# XXXS
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 32 --batch_size_ppo 256 --beta 0.01 --config_actor_deep 1 --config_actor_dim 128 --config_actor_mlp_dim 256 --config_critic_deep 1 --config_critic_dim 128 --config_critic_mlp_dim 256

# L
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 32 --batch_size_ppo 256 --beta 0.01 --config_actor_deep 4 --config_actor_dim 1024 --config_actor_mlp_dim 2048 --config_critic_deep 4 --config_critic_dim 1024 --config_critic_mlp_dim 2048 --lr_actor 0.00001 --lr_critic 0.00001

# XL
# accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=4 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 32 --batch_size_ppo 256 --beta 0.01 --config_actor_deep 8 --config_actor_dim 1024 --config_actor_mlp_dim 2048 --config_critic_deep 8 --config_critic_dim 1024 --config_critic_mlp_dim 2048 --lr_actor 0.00001 --lr_critic 0.00001

# XXL
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=8 Train.py --n 10 --m 10 --h 10 --n_epoch 80000 --batch_size_envs 16 --batch_size_ppo 128 --beta 0.01 --config_actor_deep 20 --config_actor_dim 2048 --config_actor_mlp_dim 8192 --config_actor_n_head 32 --config_critic_deep 8 --config_critic_dim 1024 --config_critic_mlp_dim 2048 --config_critic_n_head 16 --lr_actor 0.00001 --lr_critic 0.000005 