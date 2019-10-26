# dqn_pyrep_lia


#### Deep Reinforcement Learning implementation using [PyRep API](https://github.com/stepjam/PyRep) for V-Rep Simulator
> Harder, Better, **FASTER**, Stronger

---
## WIP - running some parameters testing
---

**First Things First**: Follow Pyrep instructions and make sure the pyrep Python lib is working.



Please, check Run.py to see the **argparse** to check how to run the code.

Here are just some of them:

```python

parser.add_argument('--ep',             metavar='int',   type=int,   help='Number of episodes to be executed',  default=DEFAULT_EPISODES)
parser.add_argument('--steps',          metavar='int',   type=int,   help='Number of steps to each episode',    default=DEFAULT_STEPS)
parser.add_argument('--epochs',         metavar='int',   type=int,   help='Epochs for each model.fit() call',   default=DEFAULT_EPOCHS)
parser.add_argument('--epsilon',        metavar='float', type=float, help='Random policy factor',                      required=True)
parser.add_argument('--replay_size',    metavar='int',   type=int,   help='Maximum batch size for the replay fase',    required=True)
parser.add_argument('--memory_size',    metavar='int',   type=int,   help='Memory length for the agent',               required=True)
parser.add_argument('--model',          metavar='string',type=str,   help='Model type',                                required=True)
parser.add_argument('--not_render',  help='Render (False) or not (True) the environment', action='store_true', default=False)
parser.add_argument('--debug',  help='Debug or not', action='store_true', default=False)

```

##### Feel free to add and test new models

Email-me or open an issue if have any doubts.
