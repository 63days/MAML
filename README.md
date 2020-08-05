# Model-Agnostic Meta-Learning: Pytorch Implementation
An Pytorch Implementation of the __"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks(ICML'17)"__ (a.k.a maml)


Model-Agnostic means model-independent. This meta-learning algorithm can applied to any model trained with gradient descent, including classification, regression, and reinforcement learning. 
The goal of meta-learning is to train a model using only a small number of training samples.  
In this repo, I trained a neural network with 2 hidden layers to solve sinusoid problem.

## Results
K=10  
![image](./res/graph(k=10).png)  
pre loss: 15.244 | post loss: 5.391 

## To train
`python3 main.py`

## To test
`python3 main.py --test`

## Reference
[MAML](https://arxiv.org/pdf/1703.03400.pdf)
