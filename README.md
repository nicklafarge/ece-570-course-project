# ECE 570 Course Project (Fall 2020)
**Title**: Twin-Delayed Deep Deterministic Policy Gradient (TD3)\
**Author**: Nick LaFarge


## Experiment Instructions

The code entry point is `runner.py`. At the bottom of this script, there are several options for how to configure the runner. This lets the user specify which agent (DDPG or TD3), and which OpenAI Gym environment. It also allows the user to sepcify whether or not we wish to train a new agent, or load a trained model for testing. These options can be located in the `if __name__ == '__main__':` section at the bottom of `runner.py`.

Experiments were run using Python 3.8. The specific package versions are listed in `requirements.txt`. Some additional operating system-specific installation instructions may be required to run certain OpenAI gym environments (in particular MuJoCo environments require a license and local installation). The specific install steps are OS- specific, and well documented online.

# Code Origins

TD3 and DDPG are reimplemented versions of the OpenAI Spinning UP baselines. The Spinning Up versions can be located at the following hyperlinks: [TD3](https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/td3) and [DDPG](https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/ddpg). The implementation here differs its from original Spinning Up source  in several notable ways. First, the original implementation defines the model, learning algorithm, and running script all in one place. In this paradigm, each new learning algorithm must re-implement the runner script (i.e. the basic script that keeps track of agent-environment interaction and stores relevant values). For increased flexibility, I chose to decouple these three parts into their own distinct sections of code.

While the code organization and structure differs quite a bit from the SpinningUp source, the underlying algorithm remains the same and, hence, there are certain lines in common. Rather than give a line-by-line description of all of the included code, The sections below highlight the most important changes, as well as one method that I used directly from the SpinningUp source.













## Code from other repositories

No entire scripts have been used from the original repo. That said, certain lines of code and functions are caried over mostly unchanged, and are summarized as follows:

### Replay Buffer
The ReplayBuffer implementation from SpinningUp is used almost directly in this project. This replay buffer class can be found in `DDPG/utils.py` and `TD3/utils.py`, and the original implementaiton is located at the top of [SpinningUp td3/td3.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/td3/td3.py).

### Tensorflow logistics
Some tensorflow-specific code is caried over in various forms are sumamrized in the following list. I give the line numbers these refer to in the

- Placeholder definition (`TD3/td3.py` )
- Initializtion of target networks  (`TD3/td3.py` )


### Actor gradient computation

In both the DDPG and TD3 model, I compute the update step for the actor network in two different ways. The lines in question can be found in the `define_optimizer` function of `TD3/models.py` and `DDPG/models.py`.

I implemented two methods to ensure that my implementation was correct (since each way can check the other). The second method (which is not  actually applied when the opimizer is run) is based on the gradient computation in the repositories created by [Google Research](https://github.com/google-research/google-research/blob/master/dac/ddpg_td3.py) and [TensorLearn](https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_TD3.py). Inline comments describing this usage can be seen starting at `TD3/models.py Line 155`.



## Target Network Updating
I used the SpinningUp implementations method for initializing and updating target network values using polyak averaging. This is found in `TD3/td3.py` Lines 272-278, modified from [SpinningUp td3/td3.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/td3/td3.py) Lines 205-210.







## Modified Code


### Tensorflow Networks
I modified how the original implementation handles network creation. Below I describe my original code dealing with Actor and Critic model classes. Here is specifically refer to the tensorflow code to define the networks themselves. Network creation (`TD3/modles.py` Lines 65-68, 138-141) is modified from [SpinningUp td3/core.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/td3/core.py) ( Lines 12-14, 28-38).

### Tensorflow Version
The SpinningUp implementation is based on tensorflow 1, and I added some modifications to allow the models to run using tensorflow 2.

### Optimization Step
I made some modifications to the minibatch sampling (`TD3/td3.py` Lines 212-233). This is modified from [SpinningUp td3/td3.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/td3/td3.py) (Lines 273-288)

### Minor implementation Details

- Used a similar method to keep track of a step counter compared to the initial replay steps and the uniform random policy limit.
- I changed how the uniform random action is computed during the initial time segments  (`TD3/td3.py` Line 252-258)










## Original Code

The original code files are summarized below.

### Agent Classes
I define my agent as an object, rather than a function. Creating an abstaract agent base class allows me to implement different RL methods using the same agent blueprint. The runner script knows how to interact with this "base agent", and therefore can be run using any type of RL scheme that is implemented using thi architecture. The abstract base class is located in `agent_base.py`, and is extended to create `DdpgAgent` and `Td3Agent` in `TD3/td3.py` and `DDPG/DDPG.py`. This is meant to be general by ensuring each subclass implement `on_t_update` (called after every agent-env interation) and `on_episode_complete` (called after an episode is complete). That way, each distint method can implement those methods in any way they want, without having to change the runner script itself.

### Model Classes
The models here are implemented as class structures to make it easier to create multiple identical models. The SpinningUp version relies on a function to create multi-layer perceptrons, but I thought it would be more easily understood as explicit classes to see the explicit network structure for both actor and critic. My framework defines each model as a class, located in `TD3/models.py` and `DDPG/models.py`. Each model contains a `build_network` function to create the corresponding network, and a `define_optimizer` function to define how this particular network should be optimized (i.e the loss fucntion). In the SpinningUp Version, these definitions are contained directly in the `__init__` function of the `td3` function, where networks are created by an `actor_critic` method. I think that having each model defined as a class, where it provides its own optimizer, it musch more readable, because you don't have to keep track of which variables correspond to which model as much.

### Runner
I chose to implement my own runner scrip that is not based on the SpinningUp source code. My runner is located in `runner.py`. I separated my runner out in its own file so that the DDPG and TD3 agents could be easily interchanged, and all running code exists in one common place. This is very different than the original version where each algorithm has its own running mechanism. In implementing a new runner,
one important difference between the original and reimplemented versions are that training is now conducted based on episodes, whereas in the original source, the learning process entirely formulated around a running counter of agent-environment interactions, and episodes are not at the forefront of the runner script. Data storage methods are different as well (keeping track of reward, actions, states, etc). The SpinningUp version is definitely more powerful, but I tended to favor simplicity in this implementatino to make sure all parts of the code are easily understood.


## Network saving and loading
I cimplemented a different approach to saving/loading networks, and added the ability to load the network at a particular save point (to evaluate how learning occurs over time). This saving code can be found in `agent_base.py`.









# Datasets
Reinforcement Learning does not involved pre-computed datasets. However, learning environments provided by OpenAI are used. The OpenAI "Gym" contains numerous environments that are widely used in RL papers. Further information about Gym environments are is available as follows

- [Home Page](https://gym.openai.com/)
- [Env Details](https://github.com/openai/gym/wiki/Table-of-environments)

Note that some environments require installing third-party software to function. In particular, the included code can be run onenvironments built off Box2D or MuJoCo engines. Installation instructions vary across platforms, and Mujoco requires a software license to function (free for students, and free 30 day trial available).
