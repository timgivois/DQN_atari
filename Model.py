import numpy as np
import tensorflow as tf


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


class Model:
    def __init__(self, actions, gamma=.1, epsilon=1, epsilon_decay=.9995, min_epsilon=.1, session=None, learning_rate=.1):

        self.actions = [x for x in range(len(actions))]
        # exploit vs explore value
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # deep learning neural network attributes
        self.gamma = gamma
        self.actions_made = []
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.model = self._build_model()


        if session is None:
            self.session = tf.InteractiveSession()
        else:
            self.session = session
        self.session.run(tf.global_variables_initializer())

    def _build_model(self):
        # first layer
        self.xT = tf.placeholder(tf.uint8, shape=(None, 82, 75, 3), name="x")

        self.yT = tf.placeholder(tf.float32, shape=(None), name="y")

        self.actionsT = tf.placeholder(tf.int32, shape=(None), name="actions")

        # x_normalized = tf.to_float(x) / 255.0
        X = tf.to_float(self.xT) / 255.0
        batch_size = tf.shape(X)[0]

        # 3 conv networks
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # 2 fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        predictions = tf.contrib.layers.fully_connected(fc1, len(self.actions))

        # optimizer for the nn
        gather_indices = tf.range(batch_size) * tf.shape(predictions)[1] + self.actionsT
        self.action_predictions = tf.gather(tf.reshape(predictions, [-1]), gather_indices)
        
        self.losses = tf.squared_difference(self.yT, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_op = optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())



        return predictions

    def set_session(self, session):
        self.session = session

    def predict(self, values):
        return self.session.run(self.model, feed_dict={self.xT: np.array(values).reshape(1, 82, 75, 3)})[0]

    def update(self, s, a, y):
        states_reshaped = np.array(s).reshape(len(s), 82, 75, 3)
        feed_dict = { self.xT: states_reshaped, self.yT: y, self.actionsT: a }

        loss, _ = self.session.run(
            [self.loss, self.train_op],
            feed_dict=feed_dict)

        return loss

    def explore_or_exploit(self, state):

        if np.random.binomial(1, self.epsilon):
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.predict(state)[0])

        return action

    def save_current_state(self, state, action, reward, next_state, done):
        self.actions_made.append([state, action, reward, next_state, done])

    def fit_nn(self):
        batch_size = int(len(self.actions_made)*.1) # 10% of actions made
        indexes = np.random.randint(0, len(self.actions_made), batch_size)
        actions_sampled = [self.actions_made[x] for x in indexes]
        states = []
        actionsT = []
        targets = []

        for state, action, reward, next_state, done in actions_sampled:
            target = reward
            if not done:
                target = reward + self.gamma * np.argmax(self.predict(next_state))
            
            states.append(state)
            actionsT.append(action)
            targets.append(target)
            
        self.update(states, actionsT, targets)

    def preprocess_image(self, image):
        # crop and downsample image
        new_image = image[31:195:2, 5:155:2]
        # change image colors
        #new_image = rgb2gray(new_image)
        # walls should be 142 in rgb and points should be 74
        new_image[np.logical_and(new_image[:, :] != 0, new_image[:, :] != 142)] = 74

        return new_image

    def run(self, games, env):
        max_reward = 0
        max_game = 0
        max_actions_made = 0
        for game in range(games):
            total_reward = 0
            state = self.preprocess_image(env.reset())
            while True:
                # get the next action
                action = self.explore_or_exploit(state)

                # act
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = self.preprocess_image(next_state)
                self.save_current_state(state, action, total_reward, self.preprocess_image(next_state), done)
                # get next_state

                # if finished, tell me it finished
                if done:
                    if total_reward >= max_reward:
                        max_game = game
                        max_actions_made = len(self.actions_made)
                        max_reward = total_reward
                    print('game: {}/{}, actions_made: {}, score: {}'.format(game, games, len(self.actions_made), total_reward))
                    break

            self.fit_nn()
            self.actions_made = []
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.min_epsilon else self.min_epsilon

        print("\n\nBest game {}, actions_made: {}, score {}".format(max_game, max_actions_made, max_reward))
