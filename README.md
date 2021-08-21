# Appearance-Motion Memory Consistency Network for Video Anomaly Detection
This repo is the official open source of Appearance-Motion Memory Consistency Network for Video Anomaly Detection
, AAAI 2021 by Ruichu Cai, Hao Zhang, Wen Liu,  Shenghua Gao,  Zhifeng Hao.  If you have any questions about our work, please contact me via the following email: haodotzhang@gmail.com. (Note: Since I am very busy on weekdays, I will reply emails on weekends and hope to get your understanding. Thank you!)

### 1. Prepare

```
# 1. Set up the environment
conda env create -f /your_path_to/environment.yaml


# 2. Configure some parameters
# (1). Code/main/params/const_params.py 
root_dir = "/your_path_to/ammcnet_os" 
dataset_dir = "your_root_path_dataset"
data_dir_gt = "your_root_path_dataset"
flow_model_path = "your_path"
cur_goal_tmp = "/your_path_to/ammcnet_os/log"

# (2). Code/main/eval_metric.py 
DATA_DIR = "your_dataset_root_dir" (same as dataset_dir)
```

### 2. Download datasets

```
1. avenue/ped2/shanghaitech
2. the optical flow of above datasets

We will upload the dataset as soon as possible...
```

### 3. Run

```
# ped2
python -m Code.main.run_test \
--gpu 0 \
--dataset_name ped2 

# avenue
python -m Code.main.run_test \
--gpu 0 \
--dataset_name avenue 

# shanghaitech
python -m Code.main.run_test \
--gpu 0 \
--dataset_name shanghaitech

```

### 4. auc_result

![ped2](./img/ped2.png)

![avenue](./img/avenue.png)

![shanghaitech](./img/shanghaitech.png)
