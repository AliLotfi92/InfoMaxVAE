# InfoMax Variational Autoencoder
[Learning Representaion by Maximizaing Mutual Information in Variational Autoencoders](https://ieeexplore.ieee.org/abstract/document/9174424/) in PyTorch

![alt text](https://github.com/AliLotfi92/InfoMAXVAE/blob/master/assets/png3.png)

### Requirements
- Python 3
- PyTorch 0.4
- matplotlib
- scikit-learn
- install requirements via ```
pip install -r requirements.txt``` 

### How to run
```bash
python main.py \
--method infomax \
--model mlp \
--dim 128 \
--num_iter 500 \
--batch_size 64 \
--gamma 20 \
--alpha 1 \
--dataset cifar10 \
--trial 1
```
Learning Representations by Maximizing Mutual Information in Variational Autoencoders
https://arxiv.org/abs/1912.13361


### Results for PixelCNN setup
![alt text](https://github.com/AliLotfi92/InfoMAXVAE/blob/master/assets/pixelvae.png)
