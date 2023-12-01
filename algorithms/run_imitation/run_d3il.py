import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = "osmesa"
import warnings

warnings.filterwarnings('ignore')

import csv
import time
import json
import os.path as osp
import tensorflow as tf
import numpy as np

from sac_models import StochasticActor, Critic, SAC
from samplers import Sampler

import logz
from utils import load_expert_trajectories, log_trajectory_statistics
from envs.envs import (ExpertInvertedPendulumEnv, AgentInvertedPendulumEnv,
                       ExpertInvertedDoublePendulumEnv, AgentInvertedDoublePendulumEnv,
                       CustomReacher2Env, TiltedCustomReacher2Env,
                       CustomReacher3Env, TiltedCustomReacher3Env,
                       ExpertHalfCheetahNCEnv, LockedLegsHalfCheetahNCEnv,
                       DMCartPoleBalanceEnv, DMCartPoleSwingUpEnv, DMPendulumEnv, DMAcrobotEnv)
from envs.maze_envs import CustomPointUMazeSize3Env, CustomAntUMazeSize3Env
from buffers import DemonstrationsReplayBuffer

from algorithms.run_imitation.d3il import Encoder, Generator, InvariantDiscriminator
from algorithms.run_imitation.d3il import TranslatedImageDiscriminator, ExpertFeatureDiscriminator
from algorithms.run_imitation.d3il import CustomReplayBuffer
from algorithms.run_imitation.d3il import D3ILModel, D3ILModelwithPolicy


# ==================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiment using D3IL with given parameters file.')
    parser.add_argument('--env_name', help="The source environment name.")
    parser.add_argument('--env_type', help="The domain difference in the target environment.", default=None)
    parser.add_argument('--exp_num', help="The seed number", type=int, default=-1)
    parser.add_argument('--exp_id', help="The experiment ID for identification", type=str)
    parser.add_argument('--gpu_id', help="The GPU ID", type=int, default=0)
    parser.add_argument('--save_pretrained_it_model', help="Save pretrained image translation model to reuse it",
                        default=False, action='store_true')
    parser.add_argument('--load_pretrained_it_model', help="Run pretrain phase only", default=False,
                        action='store_true')
    parser.add_argument('--only_pretrain', help="Run pretrain phase only", default=False, action='store_true')

    args = parser.parse_args()
    args.algo = 'd3il'
    return args


# ==================================================
def run_experiment(args):
    """
    Run D3IL
    """

    # ==================================================
    # GPU configuration
    gpu_id = args.gpu_id
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        try:
            for i in range(len(gpus)):
                tf.config.experimental.set_memory_growth(gpus[i], True)
        except RuntimeError as e:
            print(e)

    # ==================================================
    # Parameters: experiment
    algo = args.algo  # ex. d3il
    env_name = args.env_name  # ex. InvertedPendulum-v2
    env_type = args.env_type  # ex. to_colored
    exp_num = args.exp_num  # ex. 0
    exp_id = args.exp_id  # ex. yymmdd_hhmm
    if args.only_pretrain:
        version = 'ver_pretrain_model'  # feature extraction model training phase
    else:
        version = 'ver_train_policy'  # learner policy training phase
    expert_samples_location = 'expert_data'  # directory of expert data
    prior_samples_location = 'prior_data'  # directory of non-expert data
    pretrain_epochs = 50000  # The number of gradient steps for training feature extraction model
    pretrain_log_interval = 100  # The period to show progress of training feature extraction model
    pretrain_save_interval = 10000  # The period of saving the feature extraction model
    test_runs_per_epoch = 10  # The number of evaluation trajectories for each epoch
    steps_per_epoch = 10000  # The number of timesteps per epoch
    training_starts = 512

    # Parameters: feature extraction model
    c_gan_trans = 1.0  # loss coefficient: image adversarial
    c_gan_feat = 0.01  # loss coefficient: feature adversarial & feature prediction
    c_recon = 100000.0  # loss coefficient: image reconstruction
    c_cycle = 100000.0  # loss coefficient: image cycle consistency
    c_feat_mean = 1000.0  # loss coefficient: feature similarity
    c_feat_recon = 1000.0  # loss coefficient: feature reconstruction
    c_feat_reg = 0.1  # loss coefficient: feature regularization
    c_feat_cycle = 100.0  # loss coefficient: feature cycle consistency

    type_recon_loss = 'l2'
    eg_update_interval = 1  # In this implementation, 'eg_' stands for 'encoder and generator'.
    it_max_grad_norm = None  # In this implementation, 'it_' stands for 'image translation' (feature extraction model).
    it_lr = 3e-4
    it_updates = 0
    it_batch_size = 8

    # Parameters: learner / discriminator
    l_type = 'SAC'  # In this implementation, 'l_' stands for 'learner policy'.
    l_exploration_noise = 0.2
    l_learning_rate = 1e-3
    d_learning_rate = 1e-3  # In this implementation, 'd_' stands for 'discriminator for reward generation'.
    l_batch_size = 256
    d_batch_size = 128
    l_updates_per_step = 1      # 1 update for every 1 timestep
    d_updates_per_step = 0.02   # 1 update for every 50 timesteps
    l_act_delay = 1
    l_gamma = 0.99
    l_polyak = 0.995
    l_entropy_coefficient = 0.1
    l_tune_entropy_coefficient = True
    l_target_entropy = None
    l_clip_actor_gradients = False
    init_random_samples = 5000
    d_rew = 'mixed'
    d_max_grad_norm = None

    save_pretrained_it_model = args.save_pretrained_it_model  # Save the feature extraction model
    load_pretrained_it_model = args.load_pretrained_it_model  # Load the trained feature extraction model
    only_pretrain = args.only_pretrain  # Train only the feature extraction model
    if only_pretrain:
        save_pretrained_it_model = True

    # Parameters: environment
    # ================================================== Gym IP/RE/HC/UMaze
    if env_name == 'InvertedPendulum-v2':
        steps_per_epoch = 10000
        episode_limit = 1000
        d_updates = 20  # 1 update for every 50 timesteps
        l_buffer_size = 100000
        im_side = 32
        n_demos = 10000
        c_norm_de = 0
        c_norm_be = 0
        c_feat_cycle = 10.0
        se_env_name = 'InvertedPendulum-v2'
        sn_env_name = 'ExpertInvertedPendulum-v2'
        if env_type == 'to_colored':
            epochs = 20
            env = AgentInvertedPendulumEnv()
            tn_env_name = 'AgentInvertedPendulum-v2'
            exp_name = 'InvertedPendulum_to_colored/' + algo
        elif env_type == 'to_two':
            epochs = 100
            env = ExpertInvertedDoublePendulumEnv()
            tn_env_name = 'ExpertInvertedDoublePendulum-v2'
            exp_name = 'InvertedPendulum_to_two/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    elif env_name == 'InvertedDoublePendulum-v2':
        steps_per_epoch = 10000
        episode_limit = 1000
        d_updates = 20
        l_buffer_size = 100000
        im_side = 32
        n_demos = 10000
        c_norm_de = 0
        c_norm_be = 0
        c_feat_cycle = 10.0
        se_env_name = 'InvertedDoublePendulum-v2'
        sn_env_name = 'ExpertInvertedDoublePendulum-v2'
        if env_type == 'to_colored':
            epochs = 100
            c_feat_cycle = 100.0
            env = AgentInvertedDoublePendulumEnv()
            tn_env_name = 'AgentInvertedDoublePendulum-v2'
            exp_name = 'InvertedDoublePendulum_to_colored/' + algo
        elif env_type == 'to_one':
            epochs = 20
            env = ExpertInvertedPendulumEnv()
            tn_env_name = 'ExpertInvertedPendulum-v2'
            exp_name = 'InvertedDoublePendulum_to_one/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    elif env_name == 'Reacher2-v2':
        steps_per_epoch = 10000
        epochs = 100
        episode_limit = 50
        d_updates = 1
        l_buffer_size = 100000
        im_side = 48
        n_demos = 10000
        c_norm_de = 0
        c_norm_be = 0
        c_feat_cycle = 10000.0
        test_runs_per_epoch = 200
        se_env_name = 'Reacher2-v2'
        sn_env_name = 'Reacher2-v2'
        if env_type == 'to_tilted':
            env = TiltedCustomReacher2Env()
            tn_env_name = 'TiltedReacher2-v2'
            exp_name = 'Reacher2_to_tilted/' + algo
        elif env_type == 'to_three':
            env = CustomReacher3Env()
            tn_env_name = 'Reacher3-v2'
            exp_name = 'Reacher2_to_three/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    elif env_name == 'Reacher3-v2':
        steps_per_epoch = 10000
        epochs = 100
        episode_limit = 50
        d_updates = 1
        l_buffer_size = 100000
        im_side = 48
        n_demos = 10000
        c_norm_de = 0
        c_norm_be = 0
        c_feat_cycle = 10000.0
        test_runs_per_epoch = 200
        se_env_name = 'Reacher3-v2'
        sn_env_name = 'Reacher3-v2'
        if env_type == 'to_tilted':
            c_feat_cycle = 1000.0
            env = TiltedCustomReacher3Env()
            tn_env_name = 'TiltedReacher3-v2'
            exp_name = 'Reacher3_to_tilted/' + algo
        elif env_type == 'to_two':
            env = CustomReacher2Env()
            tn_env_name = 'Reacher2-v2'
            exp_name = 'Reacher3_to_two/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    elif env_name == 'HalfCheetah-v2':
        steps_per_epoch = 10000
        epochs = 100
        episode_limit = 200
        d_updates = 4
        l_buffer_size = 100000
        im_side = 64
        n_demos = 10000
        c_norm_de = 0
        c_norm_be = 0
        c_feat_cycle = 100.0
        pretrain_epochs = 200000
        se_env_name = 'HalfCheetah-v2'
        sn_env_name = 'HalfCheetah-v2'
        if env_type == 'to_locked_legs':
            env = LockedLegsHalfCheetahNCEnv()
            tn_env_name = 'LockedLegsHalfCheetah-v2'
            exp_name = 'HalfCheetah_to_locked_legs/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    elif env_name == 'PointUMaze-v2':
        steps_per_epoch = 10000
        epochs = 200
        episode_limit = 1000
        d_updates = 20
        l_buffer_size = 150000
        im_side = 64
        n_demos = 10000
        pretrain_epochs = 50000
        test_runs_per_epoch = 10
        c_norm_de = 1
        c_norm_be = 40
        c_feat_cycle = 10.0
        se_env_name = 'PointUMaze-v2'
        sn_env_name = 'PointUMaze-v2'
        if env_type == 'to_ant':
            env = CustomAntUMazeSize3Env()
            tn_env_name = 'AntUMaze-v2'
            exp_name = 'PointUMaze_to_ant/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    # ================================================== DMC
    elif env_name == 'DMCartPoleBalance':
        steps_per_epoch = 1000
        d_updates = 20
        l_buffer_size = 100000
        im_side = 32
        n_demos = 10000
        c_norm_de = 1
        c_norm_be = 20
        c_feat_cycle = 100.0
        pretrain_epochs = 50000
        se_env_name = 'DMCartPoleBalance'
        sn_env_name = 'DMCartPoleBalance'
        if env_type == 'to_cartpoleswingup':
            epochs = 300
            episode_limit = 1000
            env = DMCartPoleSwingUpEnv()
            tn_env_name = 'DMCartPoleSwingUp'
            exp_name = 'DMCartPoleBalance_to_cartpoleswingup/' + algo
        elif env_type == 'to_pendulum':
            epochs = 300
            episode_limit = 1000
            env = DMPendulumEnv()
            tn_env_name = 'DMPendulum'
            exp_name = 'DMCartPoleBalance_to_pendulum/' + algo
    elif env_name == 'DMCartPoleSwingUp':
        steps_per_epoch = 1000
        d_updates = 20
        l_buffer_size = 100000
        im_side = 32
        n_demos = 10000
        c_norm_de = 1
        c_norm_be = 20
        c_feat_cycle = 100.0
        pretrain_epochs = 50000
        se_env_name = 'DMCartPoleSwingUp'
        sn_env_name = 'DMCartPoleSwingUp'
        if env_type == 'to_cartpolebalance':
            epochs = 300
            episode_limit = 1000
            env = DMCartPoleBalanceEnv()
            tn_env_name = 'DMCartPoleBalance'
            exp_name = 'DMCartPoleSwingUp_to_cartpolebalance/' + algo
        elif env_type == 'to_pendulum':
            epochs = 300
            episode_limit = 1000
            env = DMPendulumEnv()
            tn_env_name = 'DMPendulum'
            exp_name = 'DMCartPoleSwingUp_to_pendulum/' + algo
        elif env_type == 'to_acrobot':
            steps_per_epoch = 10000
            epochs = 200
            episode_limit = 1000
            env = DMAcrobotEnv()
            tn_env_name = 'DMAcrobot'
            exp_name = 'DMCartPoleSwingUp_to_acrobot/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    elif env_name == 'DMPendulum':
        steps_per_epoch = 1000
        d_updates = 20
        l_buffer_size = 100000
        im_side = 32
        n_demos = 10000
        c_norm_de = 1
        c_norm_be = 20
        c_feat_cycle = 100.0
        pretrain_epochs = 50000
        se_env_name = 'DMPendulum'
        sn_env_name = 'DMPendulum'
        if env_type == 'to_cartpolebalance':
            epochs = 300
            episode_limit = 1000
            env = DMCartPoleBalanceEnv()
            tn_env_name = 'DMCartPoleBalance'
            exp_name = 'DMPendulum_to_cartpolebalance/' + algo
        elif env_type == 'to_cartpoleswingup':
            epochs = 300
            episode_limit = 1000
            env = DMCartPoleSwingUpEnv()
            tn_env_name = 'DMCartPoleSwingUp'
            exp_name = 'DMPendulum_to_cartpoleswingup/' + algo
        elif env_type == 'to_acrobot':
            steps_per_epoch = 10000
            epochs = 200
            episode_limit = 1000
            env = DMAcrobotEnv()
            tn_env_name = 'DMAcrobot'
            exp_name = 'DMPendulum_to_acrobot/' + algo
        else:
            raise NotImplementedError("Invalid env_type")
    else:
        raise NotImplementedError("Invalid env_name")

    # Reward only for UMaze task
    if env_name in ['PointUMaze-v2']:
        done_reward = 1
    else:
        done_reward = None

    # ==================================================
    # Demonstration buffers
    se_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
        se_env_name, expert_samples_location, visual_data=True, load_ids=True, max_demos=n_demos))
    sn_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
        sn_env_name, prior_samples_location, visual_data=True, load_ids=True, max_demos=n_demos))
    tn_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
        tn_env_name, prior_samples_location, visual_data=True, load_ids=True, max_demos=n_demos))

    expert_visual_data_shape = se_buffer.get_random_batch(1)['ims'][0].shape
    past_frames = expert_visual_data_shape[0]

    # ==================================================
    # Check and create directories
    save_base_path = './algorithms/run_imitation/results'
    if not os.path.isdir(save_base_path):
        os.makedirs(save_base_path)
    if not os.path.isdir(os.path.join(save_base_path, exp_name)):
        os.makedirs(os.path.join(save_base_path, exp_name))
    if not os.path.isdir(os.path.join(save_base_path, exp_name, version)):
        os.makedirs(os.path.join(save_base_path, exp_name, version))
    if not os.path.isdir(os.path.join(save_base_path, exp_name, version, exp_id)):
        os.makedirs(os.path.join(save_base_path, exp_name, version, exp_id))
    if not os.path.isdir(os.path.join(save_base_path, exp_name, version, exp_id, str(exp_num))):
        os.makedirs(os.path.join(save_base_path, exp_name, version, exp_id, str(exp_num)))
    else:
        raise FileExistsError("The directory (save_path) already exists!")
    save_final_path = os.path.join(save_base_path, exp_name, version, exp_id, str(exp_num))

    if save_pretrained_it_model:
        save_pretrained_model_base_path = './algorithms/run_imitation/pretrained_it_model'
        if not os.path.isdir(save_pretrained_model_base_path):
            os.makedirs(save_pretrained_model_base_path)
        if not os.path.isdir(os.path.join(save_pretrained_model_base_path, exp_name)):
            os.makedirs(os.path.join(save_pretrained_model_base_path, exp_name))
        else:
            raise FileExistsError("The directory (save_pretrained_model_path) already exists!")
        save_pretrained_model_final_path = os.path.join(save_pretrained_model_base_path, exp_name)

    if load_pretrained_it_model:
        load_pretrained_model_base_path = './algorithms/run_imitation/pretrained_it_model'
        load_pretrained_model_final_path = os.path.join(load_pretrained_model_base_path, exp_name)

    # ==================================================
    # Logger
    if exp_num == -1:
        print("\n\033[91m" + "Warning: logging is deactivated." + "\033[0m")
        logz.configure_output_dir(None, True)
    else:
        log_dir = os.path.join(save_final_path, 'log')
        logz.configure_output_dir(log_dir, True)

        # CSV
        csv_it_file = open(osp.join(save_final_path, 'progress_it.csv'), 'w', newline='')
        csv_it_fieldnames = ['epoch',
                             'total_loss_disc_trans', 'g_norm_disc_trans',
                             'total_loss_disc_feat', 'g_norm_disc_feat',
                             'total_loss_enc_e', 'g_norm_enc_e',
                             'total_loss_enc_d', 'g_norm_enc_d',
                             'total_loss_gen', 'g_norm_gen',
                             'loss_gan_trans_dd', 'loss_gan_trans_gg',
                             'loss_gan_feat_e_dd', 'loss_gan_feat_e_gg',
                             'loss_gan_feat_d_dd', 'loss_gan_feat_d_gg',
                             'loss_gan_feat_e_dd2', 'loss_gan_feat_e_gg2',
                             'loss_gan_feat_d_dd2', 'loss_gan_feat_d_gg2',
                             'loss_recon', 'loss_cycle',
                             'loss_feat_d_mean', 'loss_feat_e_mean',
                             'loss_feat_d_recon', 'loss_feat_e_recon',
                             'reg_se_enc_d', 'reg_sn_enc_d', 'reg_tn_enc_d', 'reg_tl_enc_d',
                             'reg_se_enc_e', 'reg_sn_enc_e', 'reg_tn_enc_e', 'reg_tl_enc_e',
                             'loss_feat_d_cycle', 'loss_feat_e_cycle']
        csv_it_writer = csv.DictWriter(csv_it_file, csv_it_fieldnames)
        csv_it_writer.writeheader()
        csv_it_file.flush()
        if save_pretrained_it_model:
            csv_it_file2 = open(osp.join(save_pretrained_model_final_path, 'progress_it.csv'), 'w', newline='')
            csv_it_writer2 = csv.DictWriter(csv_it_file2, csv_it_fieldnames)
            csv_it_writer2.writeheader()
            csv_it_file2.flush()

        if True:
            csv_file = open(osp.join(save_final_path, 'progress.csv'), 'w', newline='')
        else:
            csv_file = open(osp.join(save_final_path, 'progress.csv'), 'a', newline='')
        csv_fieldnames = ['Iteration', 'Steps', 'n', 'mean', 'max', 'min', 'std',
                          'mean_train', 'max_train', 'min_train', 'std_train']
        csv_writer = csv.DictWriter(csv_file, csv_fieldnames)
        if True:
            csv_writer.writeheader()
            csv_file.flush()

    # ==================================================
    # Print and save parameters
    exp_params = {
        'algo': algo,
        'exp_name': exp_name,
        'env_name': env_name,
        'env_type': env_type,
        'exp_num': exp_num,
        'exp_id': exp_id,
        'version': version,
        'epochs': epochs,
        'episode_limit': episode_limit,
        'pretrain_epochs': pretrain_epochs,
        'pretrain_log_interval': pretrain_log_interval,
        'test_runs_per_epoch': test_runs_per_epoch,
        'steps_per_epoch': steps_per_epoch,
        'training_starts': training_starts,
        'gpu_id': gpu_id,
    }

    it_params = {
        'im_side': im_side,
        'n_demos': n_demos,
        'expert_visual_data_shape': str(expert_visual_data_shape),
        'past_frames': past_frames,
        'c_gan_trans': c_gan_trans,
        'c_gan_feat': c_gan_feat,
        'c_recon': c_recon,
        'c_cycle': c_cycle,
        'c_feat_mean': c_feat_mean,
        'c_feat_recon': c_feat_recon,
        'c_feat_reg': c_feat_reg,
        'c_feat_cycle': c_feat_cycle,
        'type_recon_loss': type_recon_loss,
        'eg_update_interval': eg_update_interval,
        'it_max_grad_norm': it_max_grad_norm,
        'it_lr': it_lr,
        'it_updates': it_updates,
        'it_batch_size': it_batch_size,
        'c_norm_de': c_norm_de,
        'c_norm_be': c_norm_be,
    }

    d_params = {
        'd_batch_size': d_batch_size,
        'd_updates_per_step': d_updates_per_step,
        'd_rew': d_rew,
        'd_max_grad_norm': d_max_grad_norm,
        'd_learning_rate': d_learning_rate,
    }

    learner_params = {
        'l_type': l_type,
        'l_buffer_size': l_buffer_size,
        'l_exploration_noise': l_exploration_noise,
        'l_learning_rate': l_learning_rate,
        'l_batch_size': l_batch_size,
        'l_updates_per_step': l_updates_per_step,
        'l_act_delay': l_act_delay,
        'l_gamma': l_gamma,
        'l_polyak': l_polyak,
        'l_entropy_coefficient': l_entropy_coefficient,
        'l_tune_entropy_coefficient': l_tune_entropy_coefficient,
        'l_target_entropy': l_target_entropy,
        'l_clip_actor_gradients': l_clip_actor_gradients,
        'init_random_samples': init_random_samples
    }

    etc_params = {
        'save_pretrained_it_model': save_pretrained_it_model,
        'load_pretrained_it_model': load_pretrained_it_model,
        'only_pretrain': only_pretrain,
        'past_frames': past_frames,
    }

    params = {
        'exp': exp_params,
        'it': it_params,
        'd': d_params,
        'learner': learner_params,
        'etc_params': etc_params,
    }

    print("\n\033[96m" + "Parmeters:")
    print(json.dumps(params, indent=2, default=str) + "\033[0m\n")
    logz.save_params(params)

    # ==================================================
    # Layer: image translation model
    im_shape4 = [im_side, im_side, 3 * past_frames]
    im_shape1 = [im_side, im_side, 3]
    enc_e_filters = [16, 16, 32, 32, 64, 64]
    enc_d_filters = [16, 16, 32, 32, 64, 64]
    gen_filters = [64, 64, 32, 32, 16, 16, 3 * past_frames]
    dom_disc_hidden_units = [32, 32]
    cls_disc_hidden_units = [32, 32]
    trans_disc_hidden_units = [16, 16, 32, 32, 64, 64]
    if version == 'ver_train_policy':
        expert_disc_hidden_units = [100, 100]
    elif version == 'ver_pretrain_model':
        expert_disc_hidden_units = None
    else:
        raise ValueError('Invalid version.')
    enc_d_final_kernel_size = im_side // 4

    # Encoder (domain)
    def make_encoder_d():
        enc_d_layers = [tf.keras.layers.Reshape(im_shape1)]
        for i, filters in enumerate(enc_d_filters, start=1):
            if i > 2 and i % 2 == 1:
                enc_d_layers += [
                    tf.keras.layers.Conv2D(filters, kernel_size=3, strides=2, activation='relu', padding='same')]
            else:
                enc_d_layers += [
                    tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, activation='relu', padding='same')]
        enc_d_layers += [tf.keras.layers.AveragePooling2D(enc_d_final_kernel_size)]
        enc_d_layers += [tf.keras.layers.Reshape([-1])]
        enc_d_layers += [tf.keras.layers.Dense(8)]
        encoder = Encoder(enc_d_layers)
        return encoder

    # Encoder (task)
    def make_encoder_e():
        enc_e_layers = [tf.keras.layers.Reshape(im_shape4)]
        for i, filters in enumerate(enc_e_filters, start=1):
            if i > 2 and i % 2 == 1:
                enc_e_layers += [
                    tf.keras.layers.Conv2D(filters, kernel_size=3, strides=2, activation='relu', padding='same')]
            else:
                enc_e_layers += [
                    tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, activation='relu', padding='same')]
        enc_e_layers += [tf.keras.layers.Reshape([-1])]
        encoder = Encoder(enc_e_layers)
        return encoder

    # Generator
    def make_generator():
        gen_layers = []
        for i, filters in enumerate(gen_filters[:-1], start=1):
            if i > 2 and i % 2 == 1:
                gen_layers += [tf.keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2, activation='relu',
                                                               padding='same')]
            else:
                gen_layers += [tf.keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=1, activation='relu',
                                                               padding='same')]
        gen_layers += [tf.keras.layers.Conv2DTranspose(gen_filters[-1], kernel_size=1, padding='same')]

        generator = Generator(gen_layers, past_frames, n_input_channels=enc_e_filters[-1])
        return generator

    # Discriminator (domain independence)
    def make_dom_disc():
        dom_disc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in dom_disc_hidden_units]
        dom_disc_layers.append(tf.keras.layers.Dense(1))
        dom_disc = InvariantDiscriminator(dom_disc_layers, stab_const=1e-7)
        return dom_disc

    # Discriminator (class independence)
    def make_cls_disc():
        cls_disc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in cls_disc_hidden_units]
        cls_disc_layers.append(tf.keras.layers.Dense(1))
        cls_disc = InvariantDiscriminator(cls_disc_layers, stab_const=1e-7)
        return cls_disc

    # Discriminator (translated image)
    def make_trans_disc():
        trans_disc_layers = [tf.keras.layers.Reshape(im_shape4)]
        for i, filters in enumerate(trans_disc_hidden_units, start=1):
            if i > 2 and i % 2 == 1:
                trans_disc_layers += [
                    tf.keras.layers.Conv2D(filters, kernel_size=3, strides=2, activation='relu', padding='same')]
            else:
                trans_disc_layers += [
                    tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, activation='relu', padding='same')]
        trans_disc_layers += [tf.keras.layers.Reshape([-1])]
        trans_disc_layers += [tf.keras.layers.Dense(1)]
        trans_disc = TranslatedImageDiscriminator(trans_disc_layers, stab_const=1e-7)
        return trans_disc

    # Discriminator (expert)
    if version in ['ver_train_policy']:
        def make_expert_disc():
            expert_disc_layers = [tf.keras.layers.Dense(units, activation='relu') for units in expert_disc_hidden_units]
            expert_disc_layers.append(tf.keras.layers.Dense(1))
            expert_disc = ExpertFeatureDiscriminator(expert_disc_layers, stab_const=1e-7)
            return expert_disc
    elif version == 'ver_pretrain_model':
        def make_expert_disc():
            pass
    else:
        raise ValueError('Invalid version.')

    # ==================================================
    # Layer: learner
    action_size = env.action_space.shape[0]
    if l_type == 'SAC':
        def make_actor():
            actor = StochasticActor([tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                                     tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                                     tf.keras.layers.Dense(action_size * 2,
                                                           kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])
            return actor

        def make_critic():
            critic = Critic([tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                             tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                             tf.keras.layers.Dense(1,
                                                   kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])
            return critic

        if l_target_entropy is None:
            l_target_entropy = -1 * (np.prod(env.action_space.shape))
    else:
        raise NotImplementedError

    # ==================================================
    # Define agent
    l_optimizer = tf.keras.optimizers.Adam(l_learning_rate)
    if l_type == 'SAC':
        l_agent = SAC(make_actor=make_actor,
                      make_critic=make_critic,
                      make_critic2=make_critic,
                      actor_optimizer=l_optimizer,
                      critic_optimizer=l_optimizer,
                      gamma=l_gamma,
                      polyak=l_polyak,
                      entropy_coefficient=l_entropy_coefficient,
                      tune_entropy_coefficient=l_tune_entropy_coefficient,
                      target_entropy=l_target_entropy,
                      clip_actor_gradients=l_clip_actor_gradients, )
    else:
        raise NotImplementedError

    # ==================================================
    # Define sampler
    sampler = Sampler(env, episode_limit, init_random_samples, visual_env=True)

    # ==================================================
    # Define imitation model
    if only_pretrain:
        model = D3ILModel(None,
                          make_encoder_d,
                          make_encoder_e,
                          make_generator,
                          make_dom_disc,
                          make_cls_disc,
                          make_trans_disc,
                          make_expert_disc,
                          c_gan_trans,
                          c_gan_feat,
                          c_recon,
                          c_cycle,
                          c_feat_mean,
                          c_feat_recon,
                          c_feat_reg,
                          c_feat_cycle,
                          c_norm_de,
                          c_norm_be,
                          type_recon_loss,
                          eg_update_interval,
                          it_max_grad_norm,
                          it_lr,
                          d_rew,
                          d_max_grad_norm,
                          d_learning_rate,
                          past_frames=past_frames)
    elif version == 'ver_train_policy':
        model = D3ILModelwithPolicy(l_agent,
                                    make_encoder_d,
                                    make_encoder_e,
                                    make_generator,
                                    make_dom_disc,
                                    make_cls_disc,
                                    make_trans_disc,
                                    make_expert_disc,
                                    c_gan_trans,
                                    c_gan_feat,
                                    c_recon,
                                    c_cycle,
                                    c_feat_mean,
                                    c_feat_recon,
                                    c_feat_reg,
                                    c_feat_cycle,
                                    c_norm_de,
                                    c_norm_be,
                                    type_recon_loss,
                                    eg_update_interval,
                                    it_max_grad_norm,
                                    it_lr,
                                    d_rew,
                                    d_max_grad_norm,
                                    d_learning_rate,
                                    past_frames=past_frames)
    else:
        raise NotImplementedError

    # Build model
    model(model.reshape_input_images(se_buffer.get_random_batch(1)['ims']),
          model.reshape_input_images(sn_buffer.get_random_batch(1)['ims']),
          model.reshape_input_images(tn_buffer.get_random_batch(1)['ims']),
          model.reshape_input_images(tn_buffer.get_random_batch(1)['ims']),
          np.expand_dims((env.reset()).astype('float32'), axis=0))

    # Model summary
    model.summary_model(model.reshape_input_images(se_buffer.get_random_batch(1)['ims']))

    # ==================================================
    # Define replay buffer
    if env_name in ['PointUMaze-v2']:
        agent_buffer = CustomReplayBuffer(model, l_buffer_size, done_reward=done_reward, rew_multiplier=100.0)
    else:
        agent_buffer = CustomReplayBuffer(model, l_buffer_size)

    # ==================================================
    # TODO: Phase 1. Pretrain image translation model
    if load_pretrained_it_model:
        print("Phase 1. Pretrain image translation model")
        load_dir = osp.join(load_pretrained_model_final_path, str(pretrain_epochs), 'it_model.h5')
        print("\033[32m" + "Loading model weights to %s" % load_dir + "\033[0m")
        model.load_weights(load_dir, True)
        print("\033[96m" + "Phase 1. Done" + "\033[0m")
    elif pretrain_epochs > 0:
        print("Phase 1. Pretrain image translation model")
        if exp_num != -1:
            out_list_dict = dict()
            for fieldname in csv_it_fieldnames:
                out_list_dict[fieldname] = []

        for e in range(pretrain_epochs):
            if (e == 0) or (e + 1) % pretrain_log_interval == 0 or (e + 1) == pretrain_epochs:
                print('Epoch {}/{}'.format(e + 1, pretrain_epochs))

            # Get minibatch (shape = (batch_size, 4, W, H, 3))
            se_ims = se_buffer.get_random_batch(it_batch_size)['ims']
            sn_ims = sn_buffer.get_random_batch(it_batch_size)['ims']
            tn_ims = tn_buffer.get_random_batch(it_batch_size)['ims']
            tl_ims = tn_buffer.get_random_batch(it_batch_size)['ims']

            # Train model
            out = model.train_image_translation(se_ims, sn_ims, tn_ims, tl_ims, e)

            if exp_num != -1:
                csv_it_write_dict = dict()
                csv_it_write_dict["epoch"] = e + 1
                for fieldname in csv_it_fieldnames:
                    if fieldname in out.keys():
                        out_list_dict[fieldname].append((e + 1, out[fieldname]))
                        csv_it_write_dict[fieldname] = out[fieldname]
                csv_it_writer.writerow(csv_it_write_dict)
                csv_it_file.flush()
                if save_pretrained_it_model:
                    csv_it_writer2.writerow(csv_it_write_dict)
                    csv_it_file2.flush()

            if (e + 1) % pretrain_save_interval == 0 or (e + 1) == pretrain_epochs:
                if not os.path.isdir(os.path.join(save_final_path, str(e + 1))):
                    os.makedirs(os.path.join(save_final_path, str(e + 1)))
                if save_pretrained_it_model:
                    if not os.path.isdir(os.path.join(save_pretrained_model_final_path, str(e + 1))):
                        os.makedirs(os.path.join(save_pretrained_model_final_path, str(e + 1)))
                if save_pretrained_it_model:
                    save_dir = osp.join(save_pretrained_model_final_path, str(e + 1), 'it_model.h5')
                    print("\033[32m" + "Saving model weights to %s" % save_dir + "\033[0m")
                    model.save_weights(save_dir)

        print("\033[96m" + "Phase 1. Done" + "\033[0m")
    else:
        print("\033[91m" + "Skip: Phase 1. Pretrain image translation model" + "\033[0m")
        print("\033[91m" + "Warning. Image translation model is not pre-trained." + "\033[0m")
    print()

    # ==================================================
    # TODO: Phase 2. Training learner and/or image translation model
    if not only_pretrain:
        print("Phase 2. Training policy and/or image translation model")
        start_time = time.time()

        mean_test_returns = []
        mean_test_std = []
        steps = []
        step_counter = 0

        # Epoch 0
        if True:
            logz.log_tabular('Iteration', 0)
            logz.log_tabular('Steps', step_counter)

            traj_train = sampler.sample_test_trajectories(l_agent, l_exploration_noise, test_runs_per_epoch,
                                                          get_ims=False)
            out_train = log_trajectory_statistics(traj_train['ret'], False)
            logz.log_tabular('mean_train', out_train['mean'])
            logz.log_tabular('max_train', out_train['max'])
            logz.log_tabular('min_train', out_train['min'])
            logz.log_tabular('std_train', out_train['std'])

            # Evaluation
            print('Epoch {}/{} - total steps {}'.format(0, epochs, step_counter))
            out = sampler.evaluate(l_agent, test_runs_per_epoch, False, get_ims=False)
            mean_test_returns.append(out['mean'])
            mean_test_std.append(out['std'])
            steps.append(step_counter)
            for k, v in out.items():
                logz.log_tabular(k, v)
            logz.log_tabular("time", 0)
            logz.dump_tabular()

            if exp_num != -1:
                csv_write_dict = dict()
                csv_write_dict["Iteration"] = 0
                csv_write_dict["Steps"] = step_counter
                csv_write_dict["n"] = out['n']
                csv_write_dict["mean"] = out['mean']
                csv_write_dict["max"] = out['max']
                csv_write_dict["min"] = out['min']
                csv_write_dict["std"] = out['std']
                csv_write_dict["mean_train"] = out_train['mean']
                csv_write_dict["max_train"] = out_train['max']
                csv_write_dict["min_train"] = out_train['min']
                csv_write_dict["std_train"] = out_train['std']
                csv_writer.writerow(csv_write_dict)
                csv_file.flush()

                out_list_dict = dict()
                for fieldname in csv_it_fieldnames:
                    out_list_dict[fieldname] = []

        # Training Loop
        nn_updates = 0
        for e in range(epochs):
            train_epoch_ret = []
            while step_counter < (e + 1) * steps_per_epoch:
                traj_data = sampler.sample_trajectory(l_agent, l_exploration_noise)
                train_epoch_ret.append(traj_data['ret'])
                agent_buffer.add(traj_data)
                n = traj_data['n']
                step_counter += traj_data['n']
                print(step_counter)
                if step_counter > training_starts:
                    model.train(se_buffer=se_buffer,
                                sn_buffer=sn_buffer,
                                tn_buffer=tn_buffer,
                                agent_buffer=agent_buffer,
                                l_batch_size=l_batch_size,
                                l_updates=int(l_updates_per_step * n),
                                l_act_delay=l_act_delay,
                                d_updates=max(1, int(d_updates_per_step * n)),
                                d_batch_size=d_batch_size,
                                it_updates=it_updates,
                                it_batch_size=it_batch_size,
                                epoch=e,
                                pretrain_epochs=pretrain_epochs,
                                nn_updates=nn_updates,
                                step_counter=step_counter,
                                save_final_path=save_final_path, )
                    nn_updates += 1

                    if exp_num != -1 and it_updates > 0:
                        csv_it_write_dict = dict()
                        csv_it_write_dict["epoch"] = pretrain_epochs + nn_updates * it_updates
                        out_list_dict = dict()
                        for fieldname in csv_it_fieldnames:
                            if fieldname in out.keys():
                                out_list_dict[fieldname].append(
                                    (pretrain_epochs + nn_updates * it_updates, out[fieldname]))
                                csv_it_write_dict[fieldname] = out[fieldname]
                        csv_it_writer.writerow(csv_it_write_dict)
                        csv_it_file.flush()
                        if save_pretrained_it_model:
                            csv_it_writer2.writerow(csv_it_write_dict)
                            csv_it_file2.flush()
                        del csv_it_write_dict, out_list_dict
                del traj_data

            logz.log_tabular('Iteration', e + 1)
            logz.log_tabular('Steps', step_counter)
            out_train = log_trajectory_statistics(train_epoch_ret, False)
            logz.log_tabular('mean_train', out_train['mean'])
            logz.log_tabular('max_train', out_train['max'])
            logz.log_tabular('min_train', out_train['min'])
            logz.log_tabular('std_train', out_train['std'])

            # Evaluation
            print('Epoch {}/{} - total steps {}'.format(e + 1, epochs, step_counter))
            traj_test = sampler.sample_test_trajectories(l_agent, 0.0, test_runs_per_epoch, get_ims=False)
            out = log_trajectory_statistics(traj_test['ret'], False)
            mean_test_returns.append(out['mean'])
            mean_test_std.append(out['std'])
            steps.append(step_counter)
            for k, v in out.items():
                logz.log_tabular(k, v)
            logz.log_tabular("time", time.time() - start_time)
            logz.dump_tabular()

            if exp_num != -1:
                csv_write_dict = dict()
                csv_write_dict["Iteration"] = e + 1
                csv_write_dict["Steps"] = step_counter
                csv_write_dict["n"] = out['n']
                csv_write_dict["mean"] = out['mean']
                csv_write_dict["max"] = out['max']
                csv_write_dict["min"] = out['min']
                csv_write_dict["std"] = out['std']
                csv_write_dict["mean_train"] = out_train['mean']
                csv_write_dict["max_train"] = out_train['max']
                csv_write_dict["min_train"] = out_train['min']
                csv_write_dict["std_train"] = out_train['std']
                csv_writer.writerow(csv_write_dict)
                csv_file.flush()
                del csv_write_dict

            del traj_test, out
        print("\033[96m" + "Phase 2. Done" + "\033[0m")
    else:
        print("\033[91m" + "Skip: Phase 2. Training policy and/or image translation model" + "\033[0m")
        print("\033[91m" + "Warning. Policy is not trained." + "\033[0m")
    print()

    # ==================================================
    return model, sampler


# ==================================================
def main():
    """
    Run experiment for proposed imitation
    """

    # Parse arguments
    args = parse_arguments()

    # Run experiment
    model, sampler = run_experiment(args)

    # END
    print("Done!")


# ==================================================
if __name__ == '__main__':
    main()
