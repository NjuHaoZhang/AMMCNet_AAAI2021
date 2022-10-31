# #!/usr/bin/env bash
# ====== unet_vq_topk_res ====== #
python -m Code.main.run_train  \
--node 46 \
--gpu 0 \
--batch_size 16 \
--num_workers 16 \
--mode training \
--exp_tag 20200723_v_01 \
--helper_tag train_single \
--net_tag unet_vq_topk_res \
--loss_tag rgb_int_gdl_flow_adv_vq \
--data_type rgb \
--dataset_name avenue


# ==========================
python -m Code.main.run_train  \
--node 20 \
--gpu 3 \
--batch_size 8 \
--num_workers 8 \
--mode training \
--exp_tag 20200814_v_01 \
--helper_tag train_twostream \
--net_tag unet_vq_twostream \
--loss_tag twostream_vq \
--data_type rgb_op \
--dataset_name avenue \
--pretrain True

# --data_dir /ssd0/zhanghao

python -m Code.main.run_train  \
--node 49 \
--gpu 1 \
--batch_size 8 \
--num_workers 8 \
--mode training \
--exp_tag 20200815_v_01 \
--helper_tag train_twostream \
--net_tag unet_vq_twostream \
--loss_tag twostream_vq \
--data_type rgb_op \
--dataset_name shanghaitech \
--pretrain True

# lp = 0.1 (每次只变一个)
python -m Code.main.run_train  \
--node 18 \
--gpu 1 \
--batch_size 16 \
--num_workers 16 \
--mode training \
--exp_tag 20200805_v_02 \
--helper_tag train_single \
--net_tag unet_vq_topk_res \
--loss_tag rgb_int_gdl_flow_adv_vq \
--data_type rgb \
--dataset_name shanghaitech

# lp = 0.01
python -m Code.main.run_train  \
--node 18 \
--gpu 2 \
--batch_size 16 \
--num_workers 16 \
--mode training \
--exp_tag 20200805_v_03 \
--helper_tag train_single \
--net_tag unet_vq_topk_res \
--loss_tag rgb_int_gdl_flow_adv_vq \
--data_type rgb \
--dataset_name shanghaitech

# lam_latent = 0.1
python -m Code.main.run_train  \
--node 18 \
--gpu 3 \
--batch_size 16 \
--num_workers 16 \
--mode training \
--exp_tag 20200805_v_04 \
--helper_tag train_single \
--net_tag unet_vq_topk_res \
--loss_tag rgb_int_gdl_flow_adv_vq \
--data_type rgb \
--dataset_name shanghaitech
