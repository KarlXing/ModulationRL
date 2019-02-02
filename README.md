# Neuromodulation on Reinforcement Learning (A2C)

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Parameters 
* --env-name  
    name of environment
* --carl-wrapper  
    Compared with deepmind wrapper, carl wrapper doesn't do downsample. Besides that, in carl_wrapper, rewards are divided by the minimal positive reward of the game. For example, in MsPacman, the original rewards are 10, 100, 200, 500. The new rewards are 1, 10, 20, 50.
* --num-frames   
    number of frames for training
* --activation  
    activation function of the f1 layer of actor. 1 means tanh. 2 means relu. 3 means sigmoid
* --log-evaluation  
    log certain information during training
* --max-beta  
    maximum beta during training
* --min-beta  
    minimum beta during training
* --fixed-beta  
    the modulated beta is calculated by a modified sigmoid function. It's vertically stretched by a beta_range parameter. If fixed-beta is true, then beta_range is max_beta (maximum beta during training). If it's false, beta_range will increase from 1 to max_beta gradually in the 10% training frames. 
* --dynamic-lr   
  whether change learning rate in modulation (whether use the same beta when update the model). If dynamic_lr is False, learning rate will not be changed. Otherwise, modulated beta will be used in backpropogation. 
* --epsilon-start, --epsilon-end, --epsilon-decay   
    parameters for epsilon greedy
    
    
## Examples
1. no modulation   
python main_nomodulation.py --env-name "SpaceInvadersNoFrameskip-v0"  --num-frames 100000000  --carl-wrapper --activation 1 --log-evaluation 

2. no modulation, epsilon greedy  
python main_epsilon.py --env-name "SpaceInvadersNoFrameskip-v0"  --num-frames 100000000  --carl-wrapper --activation 1 --log-evaluation  --epsilon-start 0.9  --epsilon-end 0.05, --epsilon-decay 5000

3. modulation (modulate temperature in action selection only)  
    python main_modulation.py --env-name "SpaceInvadersNoFrameskip-v0"  --num-frames 100000000  --carl-wrapper --activation 1 --log-evaluation
    
4. modulation (modulate temperature in both action selection and updating model)  
python main_modulation.py --env-name "SpaceInvadersNoFrameskip-v0"  --num-frames 100000000  --carl-wrapper --activation 1 --log-evaluation  --dynamic-lr


