export CUDA_VISIBLE_DEVICES=0; nohup python train.py \
 --model configs/model/gain_bottleneck.yaml \
 --dataset configs/dataset/pets.yaml \
 --project test --clean --wandb &

export CUDA_VISIBLE_DEVICES=1; nohup python train.py \
 --model configs/model/gain_bottleneck_a2.yaml \
 --dataset configs/dataset/pets_a2.yaml \
 --project test --clean --wandb &

export CUDA_VISIBLE_DEVICES=2; nohup python train.py \
 --model configs/model/gain_bottleneck_b2.yaml \
 --dataset configs/dataset/pets_b2.yaml \
 --project test --clean --wandb &