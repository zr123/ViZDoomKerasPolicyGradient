# Reeinforcment Learning with Keras, Policy Gradient and Vizdoom

Implementing REINFORCE and A2C to train an agent to learn to play Vizdooms "Hostile Environment" Scenario. Some proof-of-concepting is done using gym's CartPole.

## What this is

A simple implementation of the most common policy gradient algorithms using keras. I tried to keep everything beginner-friendly by usings the frontend API of Keras as much as possible, while keeping the backend-stuff to a minimum. I also tried to keep my code as clean and readable as possible.

## What this isn't

A production-ready or efficient implementation. There is lots of room for optimiziation. Most glaringly this implementation currently performs the forward-propagation step twice, once during the simulation and once when calling `fit`. 

# Further Reading

If you are new to reeinforcement learning or are looking for different implementations you might find the following resources helpful:

* Some exellent and simple explanations of reinforcement learning and policy gradient: https://karpathy.github.io/2016/05/31/rl/
* Another, similar introduction to reinforcement learning using gym and vizdoom: https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/

# Disclaimer

`scenarios`-files are not mine and are directly taken from https://github.com/mwydmuch/ViZDoom 