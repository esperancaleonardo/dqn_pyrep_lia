import argparse
from Environment import Environment
from time import sleep
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os, tensorflow as tf
import glob
from datetime import datetime, date
from matplotlib import pyplot as plt
from Constants import *
import statistics
from Agent import Agent
################################################################################

parser = argparse.ArgumentParser(description=""" Tool for test and evaluate a trained model
                                                 step-by-step for a number os episodes.
                                                 """)

parser.add_argument('--gpu',           metavar='string',type=str,   help='GPU to be used',                           default='None')
parser.add_argument('--model_file',    metavar='string',type=str,   help='model file name to be used',               required=True)
parser.add_argument('--ep',             metavar='int',   type=int,   help='Number of episodes to be executed',  default=DEFAULT_EPISODES)
parser.add_argument('--steps',          metavar='int',   type=int,   help='Number of steps to each episode',    default=DEFAULT_STEPS)
parser.add_argument('--gamma',          metavar='float', type=float, help='Discount factor for the reward',            required=True)
parser.add_argument('--alpha',          metavar='float', type=float, help='Learning rate for the model',               required=True)
parser.add_argument('--epsilon',        metavar='float', type=float, help='Random policy factor',                      required=True)
parser.add_argument('--not_render',  help='Render (False) or not (True) the environment', action='store_true', default=False)

args = parser.parse_args()
################################################################################

if args.gpu != 'None':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    tf.logging.set_verbosity(tf.logging.ERROR)
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

################################################################################

if __name__ == '__main__':

    Env = Environment(not_render=args.not_render)
    Env.front_camera.set_resolution([INPUT_SIZE_90X,INPUT_SIZE_90X])
    Env.side_camera.set_resolution([INPUT_SIZE_90X,INPUT_SIZE_90X])
    Env.top_camera.set_resolution([INPUT_SIZE_90X,INPUT_SIZE_90X])

    Agent = Agent(model_string = '3_input', memory_size=10, batch_size=0,
                   input_dimension=INPUT_SIZE_90X, number_of_actions=NUMBER_OF_ACTIONS,
                   alpha=args.alpha, load_weights=True, file=args.model_file)

    EPSILON = args.epsilon

    for episode in range(args.ep):
        state = Env.reset_scene()
        episode_rw = 0.0
        done = 0

        for step in range(args.steps):
            action = Agent.act(state[3], EPSILON)
            vell = Agent.action_to_vel(action)
            reward, next_state = Env.do_step(vell, Agent.model_string)
            episode_rw += reward
            done = Env.done()
            if done: break

            print("EP {} STEP {} // ACTION {} RW/DISTANCE {}.3f CUMMR {}.3f DONE {}".format(episode+1, step+1, action, round(reward,2), round(episode_rw,2), done))
            sleep(0.001)

            state = next_state
