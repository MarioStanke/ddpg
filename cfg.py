import argparse

args = None

def init():
    global args
    parser = argparse.ArgumentParser(description='Deep Deterministic Policy Gradient Reinforcement Learning on Gym.')
    parser.add_argument('--seed', type=int, help='seed for environment and agent', default = 1)
    parser.add_argument('--n_episodes', type=int, help='number of episodes to train', default = 3001)
    parser.add_argument('--max_t', type=int, help='maximum time (number of steps)', default = 300)
    parser.add_argument('--weight_decay', type=float, help='regularization of critic', default = 0.0001)
    parser.add_argument('--lr_actor', type=float, help='learning rate of actor', default = 1e-4)
    parser.add_argument('--lr_critic', type=float, help='learning rate of critic', default = 1e-3)
    parser.add_argument('--buffer_size', type=int, help='maximum replay buffer size', default = 100000)
    parser.add_argument('--OU_theta', type=float, help='theta of Ornstein-Uhlenbeck noise', default = .15)
 
    args = parser.parse_args()

    print("seed = ", args.seed)
    print("n_episodes = ", args.n_episodes)
    print("max_t = ", args.max_t)
    print("weight_decay = ", args.weight_decay)
    print("lr_actor = ", args.lr_actor)
    print("lr_critic = ", args.lr_critic)
    print("buffer_size = ", args.buffer_size)
    print("OU_theta = ", args.OU_theta)
