# Model-Agnostic Meta-Learning: Pytorch Implementation
Pytorch Implementation of the [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) with k-shot sinusoid regression.

## Pseudocode
![image](https://user-images.githubusercontent.com/37788686/98809786-7f3c0680-2461-11eb-9735-ff29898376d7.png)

## Results
### K=10
<img src="./res/graph(k=10).png" width=600px>
pre loss: 15.244 | post loss: 5.391 


## To train
`python3 main.py`

## To test
`python3 main.py --test`
