from __future__ import print_function
import numpy as np
from data_manipulation import to_categorical
from optimizers import Adam
from loss_functions import SquareLoss
from layers import Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from neural_network import NeuralNetwork
from deep_q_network import DeepQNetwork


def main():
    dqn = DeepQNetwork(env_name='CartPole-v1',
                        epsilon=0.9, 
                        gamma=0.8, 
                        decay_rate=0.005, 
                        min_epsilon=0.1)

    # Model builder
    def model(n_inputs, n_outputs):    
        clf = NeuralNetwork(optimizer=Adam(), loss=SquareLoss)
        clf.add(Dense(64, input_shape=(n_inputs,)))
        clf.add(Activation('relu'))
        clf.add(Dense(n_outputs))
        return clf

    dqn.set_model(model)

    print ()
    dqn.model.summary(name="Deep Q-Network")

    dqn.train(n_epochs=500)
    dqn.play(n_epochs=100)

if __name__ == "__main__":
    main()