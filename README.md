# D3IL

This is an implementation of the D3IL algorithm proposed in the paper "Domain Adaptive Imitation Learning with Visual Observation" (accepted to NeurIPS 2023).

This implementation is built based on the code of "Domain-Robust Visual Imitation Learning with Mutual Information Constraints" (ICLR 2021). You can find the code [here](https://github.com/Aladoro/domain-robust-visual-il).



## Requirements

#### 1. Anaconda
We use Anaconda to set up the virtual environment. 

#### 2. MuJoCo
Before you install mujoco-py, you need a MuJoCo activation key, which is downloadable at https://www.roboti.us/license.html. (expires on October 18, 2031.)

#### 3. CUDA and CUDNN
We used Titan XP GPU or GeForce RTX 3090 GPU as our computing resource. 

If using Titan XP GPU, you are required to install CUDA 10.0 and CUDNN 7.6.4, TensorFlow 2.0, and TensorFlow Probability 0.8.0.

If using GeForce RTX 3090 GPU, you are required to install CUDA 11.4, CUDNN 8.2.4, TensorFlow 2.5, and TensorFlow Probability 0.12.0.



## Installation
If using Titan XP GPU, you can install the conda environment with the requirements using this command:
```
conda env create -f environment.yml
```

To activate the conda environment:
```
conda activate d3il
```

**Important.** 
If using GeForce RTX 3090 GPU, run this command **after** you install and activate the conda environment:
```
pip install tensorflow==2.5 gym==0.15.4 tensorflow-probability==0.12.0 
```

If you have trouble rendering MuJoCo environments off-screen, installing the following packages can solve the problem:
```
sudo apt install libsm6 libxext6 libxrender-dev libosmesa6-dev libgl1-mesa-glx libglfw3
```


## Expert demonstrations & Non-expert data
You can download expert demonstrations and non-expert data [here](https://drive.google.com/drive/folders/1ydCr219hhGONUv4dxF7sXfSecf_Ub1kW?usp=sharing).

Expert demonstrations should be stored in the `expert_data` directory.

Non-expert data should be stored in the `prior_data` directory.



## Training feature extraction model

This command trains the feature extraction model for "InvertedPendulum-to-colored" IL task and save the model.
```
python -m algorithms.run_imitation.run_d3il --env_name=InvertedPendulum-v2 --env_type=to_colored --exp_id=000000_0000 --gpu_id=0 --save_pretrained_it_model --only_pretrain
```

The following are the explanation of the options
- `exp_id`: the directory name where the performance results are stored
- `gpu_id`: indicate which GPU you want to use
- `save_pretrained_it_model`: save the feature extraction model
- `only_pretrain`: train only the feature extraction model


## Training policy

This command loads the feature extraction model and the image generation model for "InvertedPendulum-to-colored" IL task and train the policy in the target domain with seed 0.
```
python -m algorithms.run_imitation.run_d3il --env_name=InvertedPendulum-v2 --env_type=to_colored --exp_id=000000_0000 --exp_num=0 --gpu_id=0 --load_pretrained_it_model
```

The following are the explanation of the options
- `load_pretrained_it_model`: load the trained feature extraction model



## Troubleshooting
If you encountered these errors during installation and execution, you can try these methods.
1. `python: /builds/florianrhiem/pyGLFW/glfw-3.3.7/src/posix_thread.c:64: _glfwPlatformGetTls: Assertion 'tls->posix.allocated == 1' failed.`
    - Enter `export MUJOCO_GL=osmesa` before execution.


2. `ImportError: /hdd/home/(username)/anaconda3/envs/d3il/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found (required by /lib/x86_64-linux-gnu/libOSMesa.so.8)`
    - Please try these commands in sequence.
      - `strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCX`
      - `mv /home/(username)/anaconda3/envs/d3il/lib/libstdc++.so.6 /home/(username)/anaconda3/envs/d3il/lib/libstdc++.so.6.29`
      - `cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/(username)/anaconda3/envs/d3il/lib/libstdc++.so.6`



## Citation
```
@inproceedings{
choi2023domain,
title={Domain Adaptive Imitation Learning with Visual Observation},
author={Sungho Choi and Seungyul Han and Woojun Kim and Jongseong Chae and Whiyoung Jung and Youngchul Sung},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=GGbBXSkX3r}
}
```
