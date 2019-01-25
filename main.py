import os
import numpy as np
from SeqGAN.train import Trainer

# hyper parameters
B = 32 # batch size
T = 25 # Max length of sentence
g_E = 64 # Generator embedding size
g_H = 64 # Generator LSTM hidden size
g_lr = 1e-5
d_E = 64 # Discriminator embedding and Highway network sizes
d_H = 64 # Discriminator LSTM hidden size


d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # filter sizes for CNNs
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160] # num of filters for CNNs
d_dropout = 0.0 # Discriminator dropout ratio
d_lr = 1e-6

n_sample=16 # Number of Monte Calro Search 蒙特卡洛搜索的number
generate_samples = 200 # Number of generated sentences 生成句子的number


# Pretraining parameters
g_pre_lr = 1e-2
d_pre_lr = 1e-4
g_pre_epochs= 200
d_pre_epochs = 1

top = os.getcwd()
g_pre_weights_path = os.path.join(top, 'data', 'save', 'generator_pre.hdf5')
d_pre_weights_path = os.path.join(top, 'data', 'save', 'discriminator_pre.hdf5')
g_weights_path = os.path.join(top, 'data', 'save', 'generator.pkl')
d_weights_path = os.path.join(top, 'data', 'save', 'discriminator.hdf5')

trainer = Trainer(B, T, g_E, g_H, d_E, d_H, d_dropout,
    g_lr=g_lr, d_lr=d_lr, n_sample=n_sample, generate_samples=generate_samples)


# Pretraining for adversarial training 对抗训练的pretrain
trainer.pre_train(
    g_epochs=g_pre_epochs, d_epochs=d_pre_epochs,
    g_pre_path=g_pre_weights_path, d_pre_path=d_pre_weights_path,
    g_lr=g_pre_lr, d_lr=d_pre_lr)


trainer.load_pre_train(g_pre_weights_path, d_pre_weights_path)
trainer.reflect_pre_train()


trainer.train(steps=1, g_steps=1, head=10,
              g_weights_path=g_weights_path, d_weights_path=d_weights_path)

trainer.save(g_weights_path, d_weights_path)


trainer.load(g_weights_path,
             d_weights_path)

trainer.test()


trainer.generate_txt('test.txt', 100)

