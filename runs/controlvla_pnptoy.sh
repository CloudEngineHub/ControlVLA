#!/bin/bash

export HYDRA_FULL_ERROR=1

python train.py \
    --config-name=kvcontrol_fintune_diffusion_transformer_umi_workspace \
    task.dataset_path=example_finetune_demo/picknplace_toy.d10/picknplace_toy.d10.objectcentric.zarr.zip \
    dataloader.batch_size=32 \
    val_dataloader.batch_size=32 \
    dataloader.num_workers=16 \
    val_dataloader.num_workers=16 \
    training.num_epochs=801 \
    training.lr_warmup_steps=1000 \
    training.gradient_accumulate_every=2 \
    training.sample_every=50 \
    training.checkpoint_every=50 \
    checkpoint.topk.k=1 \
    checkpoint.save_last_ckpt=True \
    training.resume=True \
    training.resume_ckpt=./data/rgb_propri_10Hz-droid/checkpoints/latest.ckpt \
    task.ignore_proprioception=False \
    task.dataset.language_condition='pick up the green toy and place it into the blue bowl' \
    task.shape_meta.control_obs.camera0_rgb_narrow_objs.positional_embedding=sine \
    task.shape_meta.control_obs.camera0_rgb_narrow_objs.local_feature=graycnn \
    task.dataset.val_ratio=0.0 \
    task.dataset.repeat_frame_prob=0.5 \
    optimizer.lr=1.0e-4 \
    optimizer.obs_encoder_lr=3.0e-5 \
    policy.n_layer=3 \
    policy.n_head=4