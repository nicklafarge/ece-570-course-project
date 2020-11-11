# ECE 570 Course Project (Fall 2020)
**Title**: Twin-Delayed Deep Deterministic Policy Gradient (TD3)\
**Author**: Nick LaFarge


TD3 and DDPG are reimplemented versions of the OpenAI Spinning UP baselines. The Spinning Up versions can be located at the following URLS:

- TD3: https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/td3
- DDPG: https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/ddpg


## Experiment Running Instructions

The code entry point is `runner.py`. At the bottom of this script, there are several options for how to configure the runner.
This lets the user specify which agent (DDPG or TD3), and which OpenAI Gym environment. It also allows the user to sepcify
whether or not we wish to train a new agent, or load a trained model for testing. These options can be located in the
`if __name__ == '__main__':` section at the bottom of `runner.py`. 

Experiments were run using Python 3.8. The specific package versions are listed in `requirements.txt`. Some additional
operating system specific installation instructions may be required to run certain OpenAI gym environments (in particular
MuJoCo environments require a license and local installation). The specific install steps are operating system specific, 
and well documented online.

## Code Origins

### Models
The models here are implemented as class structures to make it easier to create multiple identical models.
The SpinningUp version relies on a function to create multi-layer perceptrons, but I thought it would be more easily
understood as explicit classes to see the explicit network structure for both actor and critic.

## Datasets
Reinforcement Learning does not involved pre-computed datasets. However, learning environments provided by OpenAI are used.
The OpenAI "Gym" contains numerous environments that are widely used in RL papers. Further information about Gym environments
are is available as follows

- Home Page: https://gym.openai.com/
- Env Details: https://github.com/openai/gym/wiki/Table-of-environments

Note that some environments require installing third-party software to function. In particular, the included code can be run on
environments built off Box2D or MuJoCo engines. Installation instructions vary across platforms, and Mujoco requires a software license
to function (free for students, and free 30 day trial available). 