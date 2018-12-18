import numpy as np
import tensorflow as tf

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class Model:
    def __init__(self, actions, gamma=.1, epsilon=.95, batch_size=50, epsilon_decay=.85):

        self.actions = [x for x in range(len(actions))]

        # exploit vs explore value
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # deep learning neural network attributes
        self.gamma = gamma
        self.batch_size = batch_size
        self.actions_made = []
        self.model = self._build_model()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())



    def _build_model(self):
        # first layer
        x = tf.placeholder(tf.float32, shape=(None, 82, 75, 1))
        # x_normalized = tf.to_float(x) / 255.0
        yT = tf.placeholder(tf.float32, shape=(None))
        # second layer, 32 8x8 filters with relu
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[8, 8],
            padding="valid",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # third layer 64 4x4 filters with relu
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # fifth layer, fully connected network
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        # sisxth layer, output
        y = tf.contrib.layers.fully_connected(dense, len(self.actions))

        # cost calculator
        cost = tf.reduce_mean(tf.square(yT - y))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        return y

    def predict(self):
        self.session.run()

    def explore_or_exploit(self, state):

        if np.random.binomial(1, self.epsilon):
            action = np.random.choice(self.actions)
        else:
            # TODO: predict a movement from the state
            pass

        return action

    def save_current_state(self, state, action, reward, next_state, done):
        self.actions_made.append(state, action, reward, next_state, done)

    def fit_nn(self):
        actions_sampled = np.random.choice(self.actions_made, self.batch_size)

        for state, action, reward, next_state, done in actions_sampled:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            # TODO: finish training of the model

    def preprocess_image(self, image):
        # crop and downsample image
        new_image = image[31:195:2, 5:155:2]
        # change image colors
        new_image = rgb2gray(new_image)
        # walls should be 142 in rgb and points should be 74
        new_image[np.logical_and(new_image[:, :] != 0, new_image[:, :] != 142)] = 74

        return new_image

    def run(self, games, times_in_epoch, env):
        state = self.preprocess_image(env.reset())

        for game in range(games):
            for j in range(times_in_epoch):
                # get the next action
                action = self.explore_or_exploit(state)
                # act
                next_state, reward, done, _ = env.step(action)
                # get next_state
                state = self.preprocess_image(next_state)
                # if finished, tell me it finished
                if done:
                    print('game: {}/{}, score: {}'.format(game, games, reward))
                    break

            self.fit_nn()
            self.actions_made = []
            self.epsilon = self.epsilon * self.epsilon_decay
