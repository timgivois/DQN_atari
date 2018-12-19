import tensorflow as tf
import gym

from Model import Model

env = gym.make('Breakout-v0')
env.reset()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'

config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    modelo = Model(env.unwrapped.get_action_meanings(), session)
    saver = tf.train.Saver()
    modelo.run(10000, env)
    save_path = saver.save(session, 'saved_model.ckpt')


