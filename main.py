# coding=utf-8
import json
import argparse
import matplotlib.pyplot as plt
from algo import PPO

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings", help="PPO configuration")
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.settings, 'r') as f:
        ppo_config = json.load(f)

    # ppo_config = {
    #     # env settings
    #     "env_name": "Pendulum-v0",
    #     "s_dim": 3,
    #     "a_dim": 1,
    #     # sample settings
    #     "max_episode": 1,
    #     "max_sample_step": 256,
    #     "discount": 0.9,
    #     "td": True,
    #     # training settings
    #     "epsilon": 0.2,        # for clipping
    #     "actor_lr": 0.0001,
    #     "critic_lr": 0.0002,
    #     "batch_size": 64,
    #     "max_update_step": 10,
    #     "max_iteration": 1000,
    #     "update_actor_step": 10
    # }
    ppo = PPO(ppo_config)

    max_iteration = ppo_config["max_iteration"]

    for k in range(max_iteration):  # for k = 1, 2, 3, ..., max_iteration
        print("iteration {}/{}: ".format(k + 1, max_iteration), end='')

        # 1. collect set of trajectories D_k={t_k} by running the policy pi(theta)
        # 2. compute reward-to-go R_t
        ppo.sample_data()

        # 3. compute advantage estimate A_t based on V_fei
        # 4. update the policy by maximizing the PPO-Clip objective(actor_loss)
        # 5. fit value function by regression on mean-square error
        ppo.update(k)

    plt.plot(range(1, max_iteration + 1), ppo.get_record())
    plt.xlabel("iteration")
    plt.ylabel("total reward")
    plt.savefig("result.png")
    plt.show()

    ppo.run_env()
