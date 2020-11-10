# Model-Agnostic Meta-Learning: Pytorch Implementation
An Pytorch Implementation of the __"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks(ICML'17)"__ (a.k.a MAML)

Model-Agnostic means model-independent. This meta-learning algorithm can applied to any model trained with gradient descent, including classification, regression, and reinforcement learning. A good machine learning model often requires training with a large number of samples. Humans, in contrast, learn new concepts and skills much faster and more efficiently. Kids who have seen cats and birds only a few times can quickly tell them apart. People who know how to ride a bike are likely to discover the way to ride a motorcycle fast with little or even no demonstration. Is it possible to design a machine learning model with similar properties — learning new concepts and skills fast with a few training examples? That’s essentially what meta-learning aims to solve.

In this repo, I train a neural network for 7k iterations on a dataset of sine function input/outputs with randomly sampled amplitude and phase, and then fine-tune on 10samples from a fixed amplitude and phase. Only 1-step adaptation shows that MAML is able to fit the sinusoid much more effectively. 
## Results
### K=10
<img src="./res/graph(k=10).png" width=600px>
pre loss: 15.244 | post loss: 5.391 

### Loss graph
<img src="./res/loss_final.png" width=600px>

## To train
`python3 main.py`

## To test
`python3 main.py --test`

## Reference
[Model-Agnostic Meta-Learning](https://arxiv.org/pdf/1703.03400.pdf)
