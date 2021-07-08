# Reeinforcment Learning with Keras, Policy Gradient and Vizdoom

Implementing several flavours of REINFORCE to train an agent to learn to play ViZDooms "Heath Gathering" Scenario. Some proof-of-concepting is done using gym's CartPole-v0.

## Installation

If you are using conda you can easily create an environment with all required packages by running:

`conda env create -f environment.yml`

If you arn't using conda you'll have to manually create an environment and install the required packages. Please also take note, that if you are using Windows you will have to [download ViZDoom 1.1.7 manually](https://github.com/mwydmuch/ViZDoom/releases) and unzip the archive to your `Lib/site-packages/` directory of your python environment.

## What this is

A simple implementation of the most common policy gradient algorithms using keras. I tried to keep everything beginner-friendly by usings the frontend API of Keras as much as possible, while keeping the backend-stuff to a minimum. I also tried to keep my code as clean and readable as possible.

## What this isn't

A production-ready or efficient implementation. There is lots of room for optimiziation. Most glaringly this implementation currently performs the forward-propagation step twice, once during the simulation and once for each episode of `fit`. 

## TODO's

* Refactoring common functions into a parent class to reduce duplicate code.
* Implementing Actor-Critic, A2C and A3C.
* Refactoring the unreadable History-object preparations to be more readable.
* Writing tests and bughunting.
* Investigate if the code can be rewritten to not perform the forward-propagation twice.
* Implementing and testing other ViZDoom scenarios.

# Further Reading

If you are new to reeinforcement learning or are looking for different implementations you might find the following resources helpful:

* If you are interested in reinforcement learning I highly recommend "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto
* Some exellent and simple explanations of reinforcement learning and policy gradient: https://karpathy.github.io/2016/05/31/rl/
* Another, similar introduction to reinforcement learning using gym and vizdoom: https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/
* I found this implentation also really helpful in my learning process: https://github.com/flyyufelix/VizDoom-Keras-RL

# Disclaimer

`scenarios`-files are not mine and are directly taken from https://github.com/mwydmuch/ViZDoom (MIT License)
