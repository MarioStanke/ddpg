import numpy as np
import tensorflow as tf 
from tensorflow.keras.layers import Dense

#TODO: reset parameter function, extra layer critic network?

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size, action_dim, units[400, 300], name="Actor"):
        super().__init__(name=name)
        
        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(action_dim, name="L3")

        self.max_action = max_action

        #with tf.device("/cpu:0")
        self(tf.constant(np.zeros(shape=(1,)+state_size, dtype=np.float32)))
    
    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        features = tf.nn.tanh(self.l3(features))
        return (self.max_action * features)



class Critic(tf.keras.Model):
    def __init__(self, state_size, action_size, units[400, 300], name="Critic"):
        super().__init__(name=name)
        
        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(1, name="L3")

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_size, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_size], dtype=np.float32))

        #with tf.device("/cpu:0"):
        self([dummy_state, dummy_action])
    
    def call(self, inputs):
        states, actions = inputs
        features = tf.concat([states, actions], axis=1)
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        return self.l3(features)
