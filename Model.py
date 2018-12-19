import numpy as np
import tensorflow as tf

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class Model:
    def __init__(self, actions, gamma=.1, epsilon=1, batch_size=50, epsilon_decay=.999):

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
        gather_indices = tf.range(batch_size) * tf.shape(predictions)[1] + self.actions
        action_predictions = tf.gather(tf.reshape(predictions, [-1]), gather_indices)

        losses = tf.squared_difference(self.yT, action_predictions)
        self.loss = tf.reduce_mean(losses)

        optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

        self.train_op = optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", losses),
            tf.summary.histogram("q_values_hist", predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(predictions))
        ])

        return predictions

    def predict(self, values):
        return self.session.run(self.model, feed_dict={self.xT: np.array(values).reshape(1, 82, 75, 3)})[0]

    def update(self, s, a, y):
        feed_dict = { self.xT: np.array(s).reshape(1,82,75,3), self.yT: y, self.actionsT: a }

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
        indexes = np.random.randint(0, len(self.actions_made), self.batch_size)
        actions_sampled = [self.actions_made[x] for x in indexes]

        for state, action, reward, next_state, done in actions_sampled:
            target = reward
            if not done:

                target = reward + self.gamma * np.argmax(self.predict(next_state))
            # gradient descent
            self.update(state, action, target)

    def preprocess_image(self, image):
        # crop and downsample image
        new_image = image[31:195:2, 5:155:2]
        # change image colors
        #new_image = rgb2gray(new_image)
        # walls should be 142 in rgb and points should be 74
        new_image[np.logical_and(new_image[:, :] != 0, new_image[:, :] != 142)] = 74

        return new_image

    def run(self, games, times_in_epoch, env):

        for game in range(games):
            total_reward = 0
            state = self.preprocess_image(env.reset())
            for j in range(times_in_epoch):
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
                    print('game: {}/{}, actions_made: {} score: {}'.format(game, games, len(self.actions_made), total_reward))
                    break

            self.fit_nn()
            self.actions_made = []
            self.epsilon = self.epsilon * self.epsilon_decay
