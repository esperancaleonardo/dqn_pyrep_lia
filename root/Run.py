import argparse
from Environment import Environment
from Agent import Agent
from time import sleep
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os, tensorflow as tf
import glob
from datetime import datetime
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
tf.logging.set_verbosity(tf.logging.ERROR)
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


parser = argparse.ArgumentParser(description=""" Parser for train a dqn agent learning how to reach some point
                                                 using V-REP simulator and the PyRep Wrapper for Python3 developed
                                                 by Coppelia Robots """)
parser.add_argument('--ep',             metavar='int',   type=int,   help='Number of episodes to be executed',  default=10)
parser.add_argument('--steps',          metavar='int',   type=int,   help='Number of steps to each episode',    default=100)
parser.add_argument('--epochs',         metavar='int',   type=int,   help='Epochs for each model.fit() call',   default=5)
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

args = parser.parse_args()

print(args)



################################################################################
def plot_fig(figure, title, x, y, filename, color):
    plt.figure(figure)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.plot(plot_data[filename], color)
    plt.savefig(os.path.join(os.getcwd(),filename))



def plot(plot_data):

    plot_fig(1, 'Recompensa por Episódio', 'Episódio', 'Valor Recompensa', str(args.model) + " ep_reward.png", 'r')

    plot_fig(2, 'Passos Gastos por Episódio', 'Episódio', 'Número de Passos', str(args.model) + " steps.png", 'r')

    plot_fig(3, 'MSE/LOSS por Episódio', 'Episódio', 'Valor MSE/LOSS', str(args.model) + " mse.png", 'r')

    plot_fig(4, 'Accuracy por Episódio', 'Episódio', 'Valor Accuracy', str(args.model) + " acc.png", 'r')

    plot_fig(5, 'Epsilon por Episódio', 'Episódio', 'Valor Epsilon', str(args.model) + " epsilon.png", 'r')



###################################################################################################################
###################################################################################################################
###################################################################################################################

if __name__ == '__main__':

    plot_data = {"ep_reward.png":[],
                 "mse.png":[],
                 "steps.png":[],
                 "epsilon.png":[],
                 "acc.png":[]}

    Env = Environment(not_render=args.not_render)

    if args.model == 'base':
        input_size = 64
        Env.front_camera.set_resolution([input_size,input_size])
        Env.side_camera.set_resolution([input_size,input_size])
        Env.top_camera.set_resolution([input_size,input_size])
    else:
        input_size = 90
        Env.front_camera.set_resolution([input_size,input_size])
        Env.side_camera.set_resolution([input_size,input_size])
        Env.top_camera.set_resolution([input_size,input_size])

    model_file = glob.glob('*.h5')
    if(len(model_file) == 1):
        Agent = Agent(model_string = args.model, memory_size=args.memory_size, batch_size= args.replay_size,
                       input_dimension=input_size, number_of_actions=14,
                       alpha=args.alpha, load_weights=args.load,
                       file=model_file[0])
    else:
        Agent = Agent(model_string = args.model, memory_size=args.memory_size, batch_size= args.replay_size,
                       input_dimension=input_size, number_of_actions=14,
                       alpha=args.alpha, load_weights=args.load)


    EPSILON = args.epsilon


    for episode in range(args.ep):
        state = Env.reset_scene()
        episode_rw = 0.0
        done = 0
        for step in tqdm(range(args.steps)):
            action = Agent.act(state[3], EPSILON)
            vell = Agent.action_to_vel(action)
            reward, next_state = Env.do_step(vell)
            episode_rw += reward
            done = Env.done()
            if done: break
            Agent.write_memory(state[3], action, reward, done, next_state[3])
            state = next_state

        if len(Agent.memory) >= int(Agent.BATCH_SIZE):
            evall = Agent.replay(args.gamma, args.epochs)
            now = datetime.now()
            if evall == 0:
                print("{} mse/loss --> {} accuracy --> {}".format(str(now), 0, evall.history['acc']))
                plot_data["mse.png"].append(0)
                plot_data["acc.png"].append(round(evall.history['acc'][0],6))
            else:
                print("{} mse/loss --> {} accuracy --> {}".format(str(now), evall.history['mean_squared_error'], evall.history['acc']))
                plot_data["mse.png"].append(round(evall.history['mean_squared_error'][0],6))
                plot_data["acc.png"].append(round(evall.history['acc'][0],6))


        now = datetime.now()
        print("{} {}/{} episodes //// DONE {}".format(str(now), episode+1, args.ep, True if done==1 else False))
        print("{} reward --> {}".format(str(now), episode_rw))


        plot_data["ep_reward.png"].append(episode_rw)
        plot_data["epsilon.png"].append(EPSILON)
        plot_data["steps.png"].append(step)

        if (episode+1)%args.episodes_decay==0:
            EPSILON *= args.decay

        if (episode+1)%10 == 0:
            plot(plot_data=plot_data)
            Agent.model.save_weights('model.h5')


    print(EPSILON)
    print(plot_data)

    plot(plot_data=plot_data)


    Agent.model.save_weights('model.h5')

    Env.shutdown()
