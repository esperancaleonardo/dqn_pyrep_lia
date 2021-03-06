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



todays_date = date.today()

parser = argparse.ArgumentParser(description=""" Parser for train a dqn agent learning how to reach some point
                                                 using V-REP simulator and the PyRep Wrapper for Python3 developed
                                                 by Coppelia Robots """)
parser.add_argument('--gpu',           metavar='string',type=str,   help='GPU to be used',                           required=True)
parser.add_argument('--name',           metavar='string',type=str,   help='Name to be used',                           required=True)
parser.add_argument('--ep',             metavar='int',   type=int,   help='Number of episodes to be executed',  default=DEFAULT_EPISODES)
parser.add_argument('--steps',          metavar='int',   type=int,   help='Number of steps to each episode',    default=DEFAULT_STEPS)
parser.add_argument('--epochs',         metavar='int',   type=int,   help='Epochs for each model.fit() call',   default=DEFAULT_EPOCHS)
parser.add_argument('--gamma',          metavar='float', type=float, help='Discount factor for the reward',            required=True)
parser.add_argument('--alpha',          metavar='float', type=float, help='Learning rate for the model',               required=True)
parser.add_argument('--epsilon',        metavar='float', type=float, help='Random policy factor',                      required=True)
parser.add_argument('--min_epsilon',    metavar='float', type=float, help='Minimum Epsilon value',                     required=True)
parser.add_argument('--decay',          metavar='float', type=float, help='Decay factor for epsilon value',            required=True)
parser.add_argument('--episodes_decay', metavar='int',   type=int,   help='Episodes needed for decay',                 required=True)
parser.add_argument('--replay_size',    metavar='int',   type=int,   help='Maximum batch size for the replay fase',    required=True)
parser.add_argument('--memory_size',    metavar='int',   type=int,   help='Memory length for the agent',               required=True)
parser.add_argument('--model',          metavar='string',type=str,   help='Model type',                                required=True)
parser.add_argument('--load',        help='Load previous weights for the keras model', action='store_true', default=False)
parser.add_argument('--not_render',  help='Render (False) or not (True) the environment', action='store_true', default=False)
parser.add_argument('--debug',  help='Debug or not', action='store_true', default=False)

args = parser.parse_args()

############################################################################################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
tf.logging.set_verbosity(tf.logging.ERROR)
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from Agent import Agent

################################################################################
def plot_fig(figure, title, x, y, filename, color):
    plt.figure(figure)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.plot(plot_data[filename], color)
    plt.savefig(os.path.join(os.getcwd(),filename))

def plot_mean(figure, title, x, y, filename, color):
    plt.figure(figure)
    plt.clf()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

    mean = []

    x = 0
    while  (x < len(plot_data[concat(args, '_ep_reward.png')])):
        mean.append(statistics.mean(plot_data[concat(args, '_ep_reward.png')][x:x+20]))
        x += 20

    plt.plot(mean, color)
    plt.savefig(os.path.join(os.getcwd(),filename))


def plot(plot_data):

    plot_fig(1, 'Recompensa por Episodio', 'Episodio', 'Valor Recompensa', concat(args, '_ep_reward.png'), 'r')

    plot_fig(2, 'Passos Gastos por Episodio', 'Episodio', 'Numero de Passos', concat(args, '_steps.png'), 'r')

    plot_fig(3, 'MSE/LOSS por Episodio', 'Episodio', 'Valor MSE/LOSS', concat(args, '_mse.png'), 'r')

    plot_fig(4, 'Accuracy por Episodio', 'Episodio', 'Valor Accuracy', concat(args, '_acc.png'), 'r')

    plot_fig(5, 'Epsilon por Episodio', 'Episodio', 'Valor Epsilon', concat(args, '_epsilon.png'), 'r')

    plot_mean(6, 'Recompensa Media-20 por Episodio', 'Episodio', 'Valor Recompensa', concat(args, '_mean_ep_reward.png'), 'r')

    plot_fig(7, 'Menor Distancia por Episodio', 'Episodio', 'Distancia', concat(args, '_min_dist.png'), 'r')

    plot_fig(8, 'Ultima Distancia por Episodio', 'Episodio', 'Distancia', concat(args, '_last_dist.png'), 'r')





def concat(_args, png_string):
    return str( str(_args.name) + '_' +
                str(todays_date) + '_' +
                str(_args.model) + '_' +
                str(_args.epochs) + '_' +
                str(_args.alpha) + '_' +
                str(_args.ep) + 'ep_' +
                str(_args.replay_size) + 'of' + str(_args.memory_size) + str(png_string) )


###################################################################################################################
###################################################################################################################
###################################################################################################################

if __name__ == '__main__':


    plot_data = {concat(args, '_ep_reward.png'):[],
                 concat(args, '_mse.png'):[],
                 concat(args, '_steps.png'):[],
                 concat(args, '_epsilon.png'):[],
                 concat(args, '_acc.png'):[],
                 concat(args, '_min_dist.png'):[],
                 concat(args, '_last_dist.png'):[]}

    Env = Environment(not_render=args.not_render)

    if args.model == 'base':
        Env.front_camera.set_resolution([INPUT_SIZE_64X,INPUT_SIZE_64X])
        Env.side_camera.set_resolution([INPUT_SIZE_64X,INPUT_SIZE_64X])
        Env.top_camera.set_resolution([INPUT_SIZE_64X,INPUT_SIZE_64X])

        model_file = glob.glob('*.h5')
        if(len(model_file) == 1):
            Agent = Agent(model_string = args.model, memory_size=args.memory_size, batch_size= args.replay_size,
                           input_dimension=INPUT_SIZE_64X, number_of_actions=NUMBER_OF_ACTIONS,
                           alpha=args.alpha, load_weights=args.load,
                           file=model_file[0])
        else:
            Agent = Agent(model_string = args.model, memory_size=args.memory_size, batch_size= args.replay_size,
                           input_dimension=INPUT_SIZE_64X, number_of_actions=NUMBER_OF_ACTIONS,
                           alpha=args.alpha, load_weights=args.load)

    else:
        Env.front_camera.set_resolution([INPUT_SIZE_90X,INPUT_SIZE_90X])
        Env.side_camera.set_resolution([INPUT_SIZE_90X,INPUT_SIZE_90X])
        Env.top_camera.set_resolution([INPUT_SIZE_90X,INPUT_SIZE_90X])

        model_file = glob.glob('*.h5')
        if(len(model_file) == 1):
            Agent = Agent(model_string = args.model, memory_size=args.memory_size, batch_size= args.replay_size,
                           input_dimension=INPUT_SIZE_90X, number_of_actions=NUMBER_OF_ACTIONS,
                           alpha=args.alpha, load_weights=args.load,
                           file=model_file[0])
        else:
            Agent = Agent(model_string = args.model, memory_size=args.memory_size, batch_size= args.replay_size,
                           input_dimension=INPUT_SIZE_90X, number_of_actions=NUMBER_OF_ACTIONS,
                           alpha=args.alpha, load_weights=args.load)



    EPSILON = args.epsilon


    for episode in tqdm(range(args.ep)):
        state = Env.reset_scene()
        episode_rw = 0.0
        done = 0
        Agent.min_rw = 1000
        Agent.last_rw = 0
        for step in range(args.steps):
            action = Agent.act(state[3], EPSILON)
            vell = Agent.action_to_vel(action)
            reward, next_state = Env.do_step(vell, args.model)
##############################
            #print(reward)
            Agent.last_rw = reward
            Agent.min_rw = reward if (reward < Agent.min_rw) else Agent.min_rw
##############################
            episode_rw += reward
            done = Env.done()
            if done: break
            Agent.write_memory(state[3], action, reward, done, next_state[3])
            state = next_state

        if len(Agent.memory) >= int(Agent.BATCH_SIZE):
            evall = Agent.replay(args.gamma, args.epochs)
            now = datetime.now()
            if evall == 0:
                if args.debug:
                    print("{} mse/loss --> {} accuracy --> {}".format(str(now), 0, evall.history['acc']))
                plot_data[concat(args, "_mse.png")].append(0)
                plot_data[concat(args,"_acc.png")].append(round(evall.history['acc'][0],6))
            else:
                if args.debug:
                    print("{} mse/loss --> {} accuracy --> {}".format(str(now), evall.history['mean_squared_error'], evall.history['acc']))
                plot_data[concat(args, "_mse.png")].append(round(evall.history['mean_squared_error'][0],6))
                plot_data[concat(args,"_acc.png")].append(round(evall.history['acc'][0],6))


        now = datetime.now()
        if args.debug:
            print("{} {}/{} episodes //// DONE {}".format(str(now), episode+1, args.ep, True if done==1 else False))
            print("{} reward --> {}".format(str(now), episode_rw))

        #print("{} // ({}/{}) episodes //".format(str(now), episode+1, args.ep))

        plot_data[concat(args, '_min_dist.png')].append(Agent.min_rw)
        plot_data[concat(args, '_last_dist.png')].append(Agent.last_rw)
        plot_data[concat(args, "_ep_reward.png")].append(episode_rw)
        plot_data[concat(args, "_epsilon.png")].append(EPSILON)
        plot_data[concat(args, "_steps.png")].append(step)

        if (episode+1)%args.episodes_decay==0:
            EPSILON *= args.decay

        if (episode+1)%20 == 0:
            plot(plot_data=plot_data)
            Agent.model.save_weights(concat(args, ".h5"))


    print(EPSILON)


    plot(plot_data=plot_data)


    Agent.model.save_weights(concat(args, ".h5"))

    Env.shutdown()
