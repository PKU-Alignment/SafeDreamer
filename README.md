
<br>
<p align="center">
<h1 align="center"><img align="center" width="6.5%"><strong>SafeDreamer: Safe Reinforcement Learning with World Models
</strong></h1>
  <p align="center">
    <a href='https://github.com/hdadong/' target='_blank'>Weidong Huang</a>&emsp;
    <a href='https://jijiaming.com/' target='_blank'>Jiaming Ji</a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=f1BzjccAAAAJ' target='_blank'>Chunhe Xia</a>&emsp;
    <a href='https://github.com/muchvo' target='_blank'>Borong Zhang</a>&emsp;
    <a href='https://www.yangyaodong.com/' target='_blank'>Yaodong Yang</a>&emsp;
    <br>
    Beihang University&emsp;Peking University
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2307.07176" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2307.07176-blue?">
  </a>
  <a href="https://sites.google.com/view/safedreamer" target='_blank'>
    <img src="https://img.shields.io/badge/Website-&#x1F680-green">
  </a>
  <a href="https://sites.google.com/view/safedreamer" target='_blank'>
    <img src="https://img.shields.io/badge/Model Checkpoint-&#x1F60-red">
  </a>
</p>



## 🏠 About
The deployment of Reinforcement Learning (RL) in real-world applications is constrained by its failure to satisfy safety criteria. Existing Safe Reinforcement Learning (SafeRL) methods, which rely on cost functions to enforce safety, often fail to achieve zero-cost performance in complex scenarios, especially vision-only tasks. These limitations are primarily due to model inaccuracies and inadequate sample efficiency. The integration of world models has proven effective in mitigating these shortcomings. In this work, we introduce SafeDreamer, a novel algorithm incorporating Lagrangian-based methods into world model planning processes within the superior Dreamer framework. Our method achieves nearly zero-cost performance on various tasks, spanning low-dimensional and vision-only input, within the Safety-Gymnasium benchmark, showcasing its efficacy in balancing performance and safety in RL tasks. 
<!-- ![Teaser](assets/teaser.jpg) -->
<div style="text-align: center;">
    <img src="assets/architecture-min.png" alt="Dialogue_Teaser" width=100% >
</div>

We have also open-sourced over **80+** [model checkpoints](https://huggingface.co/Weidong-Huang/SafeDreamer) for 20 tasks. Our codebase supports vector and vision observations. We hope this repository will become a valuable community resource for future research on model-based safe reinforcement learning.

## 🔥 News
- [2024-04] We have open-sourced the code and 80+ model checkpoints.
- [2024-01] SafeDreamer has been accepted for ICLR 2024. 

## 🔍 Overview

### Framework
<p align="center">
  <img src="assets/brain-min.png" align="center" width="100%">
</p>
The Architecture of SafeDreamer. (a) illustrates all components of SafeDreamer, which distinguishes costs as safety indicators from rewards and balances them using the Lagrangian method and a safe planner. The OSRP (b) and OSRP-Lag (c) variants execute online safety-reward planning (OSRP) within the world models for action generation. OSRP-Lag integrates online planning with the Lagrangian approach to balance long-term rewards and costs. The BSRP-Lag variant of SafeDreamer (d) employs background safety-reward planning (BSRP) via the Lagrangian method within the world models to update a safe actor. 


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
safedreamer,
title={SafeDreamer: Safe Reinforcement Learning with World Models},
author={Weidong Huang and Jiaming Ji and Borong Zhang and Chunhe Xia and Yaodong Yang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tsE5HLYtYg}
}
```

## Instructions

### Step0: Git clone
```sh
git clone https://github.com/PKU-Alignment/SafeDreamer.git
cd SafeDreamer
```

### Step1: Check version of CUDA and CUDNN (if use GPU)
Due to the strong dependency of JAX on CUDA and cuDNN, it is essential to ensure that the versions are compatible to run the code successfully. Before installing JAX, it is recommended to carefully check the CUDA and cuDNN versions installed on your machine. Here are some methods we provide for checking the versions:

1. Checking CUDA version:
- Use the command `nvcc --version` in the terminal to check the installed CUDA version.

2. Checking cuDNN version:
- Check the version by examining the file names or metadata in the cuDNN installation directory 'cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2'.
- Or you can also use torch to check the CUDNN version 'python3 -c 'import torch;cudnn_version = torch.backends.cudnn.version();print(f"CUDNN Version: {cudnn_version}");print(torch.version.cuda)'

It is crucial to ensure that the installed CUDA and cuDNN versions are compatible with the specific version of JAX you intend to install.
### Step2: Install jax
Here is some subjections for install jax, the new manipulation should be found in [jax](https://github.com/google/jax) documentation. we tested our code in the 0.3.25 version of jax.

### 
```sh
conda create -n example python=3.8
conda activate example
pip install --upgrade pip
pip install jax==0.3.25
pip install jax-jumpy==1.0.0
# for gpu
pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# for cpu
pip install jaxlib==0.3.25
```

### Step3: Install Other Dependencies
```sh
pip install -r requirements.txt
```

### Step4: Install Safetygymnasium
```sh
git clone https://github.com/PKU-Alignment/safety-gymnasium.git
cd safety-gymnasium
pip install -e .
cd ..
```

### Step5: Evaluation using Checkpoints
You can download the checkpoint from [Hugging Face](https://huggingface.co/Weidong-Huang/SafeDreamer/tree/main) and then run it locally without training from scratch. If you're looking to see if the code can run correctly, we recommend you download [the checkpoints of SafeDreamer(OSRP-Vector)](https://huggingface.co/Weidong-Huang/SafeDreamer/tree/main/safedreamer_osrp_vector), as it has smaller size:



|       Algorithm       | Size |  Checkpoint Link  |                                                                                                                                                                                                                | 
| :----------------: | :--: | :-------------: | ----------------------------------------------------------------------------------------------------------------------------- |
|SafeDreamer(BSRP-Lag)| 392MB | [Hugging Face](https://huggingface.co/Weidong-Huang/SafeDreamer/tree/main/safedreamer_bsrplag) 
|SafeDreamer(OSRP-Lag)| 392MB | [Hugging Face](https://huggingface.co/Weidong-Huang/SafeDreamer/tree/main/safedreamer_osrplag) 
|SafeDreamer(OSRP)| 392MB | [Hugging Face](https://huggingface.co/Weidong-Huang/SafeDreamer/tree/main/safedreamer_osrp) 
|SafeDreamer(OSRP-Vector)| 26.6MB | [Hugging Face](https://huggingface.co/Weidong-Huang/SafeDreamer/tree/main/safedreamer_osrp_vector) 
|Unsafe-DreamerV3| 340MB | [Hugging Face](https://huggingface.co/Weidong-Huang/SafeDreamer/tree/main/unsafe_dreamerv3) 

```sh
# Background Safety-Reward Planning with Lagrangian (BSRP-Lag):
python SafeDreamer/train.py --configs bsrp_lag --method bsrp_lag --run.script eval_only --run.from_checkpoint /xxx/checkpoint.ckpt  --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.steps 10000

# Online Safety-Reward Planning with Lagrangian (OSRP-Lag):
python  SafeDreamer/train.py --configs osrp_lag --method osrp_lag --run.script eval_only --run.from_checkpoint /xxx/checkpoint.ckpt --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.steps 10000 --pid.init_penalty 0.1

# Online Safety-Reward Planning (OSRP):
python  SafeDreamer/train.py --configs osrp --method osrp --run.script eval_only --run.from_checkpoint /xxx/checkpoint.ckpt --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.steps 10000

# Online Safety-Reward Planning (OSRP) for low-dimensional input:
python  SafeDreamer/train.py --configs osrp_vector --method osrp --run.script eval_only --run.from_checkpoint /xxx/checkpoint.ckpt --task safetygymcoor_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.steps 10000

```

where checkpoint_path is '/xxx/xxx.ckpt'. If you use cpu, you should change the "--jax.logical_gpus 0" to "--jax.platform cpu".


### Step6: Training from Scratch
```sh
# For cpu:
python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.platform cpu

# For gpu:
# Online Safety-Reward Planning (OSRP):
python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0

# Online Safety-Reward Planning with Lagrangian (OSRP-Lag):
python SafeDreamer/train.py --configs osrp_lag --method osrp_lag --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0

# Background Safety-Reward Planning with Lagrangian (BSRP-Lag):
python SafeDreamer/train.py --configs bsrp_lag --method bsrp_lag --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0

# Online Safety-Reward Planning (OSRP) for low-dimensional input:
python SafeDreamer/train.py --configs osrp_vector --method osrp_vector --task safetygymcoor_SafetyPointGoal1-v0 --jax.logical_gpus 0

```

## Tips

- All configuration options are documented in `configs.yaml`, and you have the ability to override them through the command line.
- If you encounter CUDA errors, it is recommended to scroll up through the error messages, as the root cause is often an issue that occurred earlier, such as running out of memory or having incompatible versions of JAX and CUDA.
- To customize the GPU memory requirement, you can modify the `os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']` variable in the `jaxagent.py`. This allows you to adjust the memory allocation according to your specific needs.


## 📄 License
SafeDreamer is released under Apache License 2.0.



## 👏 Acknowledgements
- [DreamerV3](https://github.com/danijar/dreamerv3): Our codebase is built upon DreamerV3.
