import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
from train_expert import train_expert
from collect_expert_data import collect_expert_data
from collect_prior_data import collect_prior_data
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# ==================================================
def main():
    """
    Collect expert or prior data for imitation learning.

    Run example:
    python -m algorithms.collect_data_for_imitation --demo_type=expert --env_name=InvertedPendulum-v2
    python -m algorithms.collect_data_for_imitation --demo_type=expert --env_name=InvertedPendulum-v2 --save_expert
    python -m algorithms.collect_data_for_imitation --demo_type=expert --env_name=InvertedPendulum-v2 --load_expert
    python -m algorithms.collect_data_for_imitation --demo_type=prior --realm_name=InvertedPendulum
    """

    # ==================================================
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument('--demo_type', help="Collect expert/prior data.", choices=['expert', 'prior'])
    parser.add_argument('--env_name', help="Environment name (if demo_type=='expert').", default=None)
    parser.add_argument('--realm_name', help="Realm name (if demo_type=='prior').", default=None)
    parser.add_argument('--demo_timesteps', help="The number of timesteps for each demo", type=int, default=10000)
    parser.add_argument('--save_expert', help="Save expert model.", default=False, action='store_true')
    parser.add_argument('--load_expert', help="Load expert model.", default=False, action='store_true')
    args = parser.parse_args()

    # ==================================================
    # Check arguments
    demo_type = args.demo_type
    env_name = args.env_name
    realm_name = args.realm_name
    demo_timesteps = args.demo_timesteps
    save_expert = args.save_expert
    save_expert_path = './sac_expert/' + env_name + '/model' if save_expert else None
    load_expert = args.load_expert
    load_expert_path = './sac_expert/' + env_name + '/model' if load_expert else None

    if save_expert and load_expert:
        raise ValueError("Both 'save_expert' and 'load_expert' cannot be active.")

    print("\033[96m", end='')
    print("Demo type                : {}".format(demo_type))
    if demo_type == 'expert':
        print("Env name                 : {}".format(env_name))
    elif demo_type == 'prior':
        print("Realm name               : {}".format(realm_name))
    print("Demo timesteps           : {}".format(demo_timesteps))
    if demo_type == 'expert':
        print("RL algorithm for expert  : {}".format('SAC'))
        if save_expert:
            print("Save expert path         : {}".format(save_expert_path))
        if load_expert:
            print("Load expert path         : {}".format(load_expert_path))
    print("\033[0m\n")

    # ==================================================
    # Collect expert/prior data for imitation learning
    if demo_type == 'expert':
        # ==================================================
        if env_name is None:
            raise KeyError("Argument 'env_name' is None. Specify the environment name.")

        # ==================================================
        if not load_expert:
            # Train expert agent
            expert_agent = train_expert(env_name=env_name)

            # Save expert agent
            if save_expert:
                expert_agent.save_weights(filepath=save_expert_path)
        else:
            # Load expert agent
            from algorithms.collect_demo.utils_demo import make_dummy_expert
            expert_agent = make_dummy_expert(env_name=env_name)
            expert_agent.load_weights(filepath=load_expert_path)

        # Collect expert trajectories
        collect_expert_data(agent=expert_agent,
                            env_name=env_name,
                            max_timesteps=demo_timesteps,
                            expert_samples_location='expert_data')
    elif demo_type == 'prior':
        # ==================================================
        if realm_name is None:
            raise KeyError("Argument 'realm_name' is None. Specify the realm name.")

        # ==================================================
        # Collect prior data
        collect_prior_data(realm_name=realm_name,
                           max_timesteps=demo_timesteps,
                           prior_samples_location='prior_data')
    else:
        raise KeyError("demo_type must be either 'expert' or 'prior'.")


# ==================================================
if __name__ == '__main__':
    main()
