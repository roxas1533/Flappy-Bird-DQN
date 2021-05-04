import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python import he_normal
from tensorflow_core.python.keras import Input, Model
from tensorflow_core.python.keras.backend import mean, expand_dims
from tensorflow_core.python.keras.layers import Concatenate, Lambda
from tensorflow_core.python.keras.losses import Huber

import Flappy
from SumTree import SumTree

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.compat.v1.disable_eager_execution()


class QNetWork:
    def __init__(self, state_size, action_size):
        inputs = Input(shape=state_size)
        x = Conv2D(32, kernel_size=(8, 8), padding='same', strides=4, activation='relu', data_format="channels_last",
                   kernel_initializer=he_normal())(inputs)
        x = Conv2D(64, kernel_size=(4, 4), padding='same', strides=2, activation='relu', data_format="channels_last",
                   kernel_initializer=he_normal())(x)
        x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=1, activation='relu', data_format="channels_last",
                   kernel_initializer=he_normal())(x)
        x = Flatten()(x)
        v = Dense(512, activation='relu', kernel_initializer=he_normal())(x)
        v = Dense(1)(v)

        ad = Dense(512, activation='relu')(x)
        ad = Dense(action_size)(ad)
        c = Concatenate()([v, ad])
        c = Lambda(lambda a: expand_dims(a[:, 0], -1) + a[:, 1:] - mean(a[:, 1:], axis=1, keepdims=True),
                   output_shape=(action_size,))(c)

        # outputs = Dense(action_size, activation='linear', kernel_initializer=he_normal())(x)
        self.model = Model(inputs=inputs, outputs=c)
        self.model.compile(loss=Huber(), optimizer=Adam(lr=0.001))


class _head_wrapper():
    def __init__(self, data):
        self.d = data

    def __eq__(self, other):
        return True


class Memory:
    def __init__(self, memory_size, alpha):
        self.buffer = []
        self.capacity = memory_size
        self.tree = SumTree(memory_size)
        self.alpha = alpha
        self.beta_initial = 0.0
        self.beta_steps = 200000
        self.max_priority = 1

    def add(self, experience):
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')
        beta = self.beta_initial + (1 - self.beta_initial) * total_step / self.beta_steps

        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section * i + random.random() * section
            (idx, priority, experience) = self.tree.get(r)

            indexes.append(idx)
            batchs.append(experience)
            weights[i] = (self.total() * priority / total) ** (-beta)
        weights = weights / weights.max()
        return indexes, batchs, weights

    def update(self, index, experience, td_error):
        priority = (abs(td_error) + 0.0001) ** self.alpha
        self.tree.update(index, priority)

        if self.max_priority < priority:
            self.max_priority = priority

    def total(self):
        return self.tree.total()

    def __len__(self):
        return len(self.buffer)


modelLoad = True

E_START = 0.1
E_STOP = 0.0001
E_DECAY_RATE = 0.00001
BATCH_SIZE = 32
REPLAY_MEMORY = 50000
env = Flappy.FlappyClass()
state_size = env.observation_space.shape
action_size = env.action_space.n
main_qn = QNetWork(state_size=state_size, action_size=action_size)
target_qn = QNetWork(state_size=state_size, action_size=action_size)
memory = Memory(REPLAY_MEMORY, 0.6)
epsilon = E_START
episode = 0
total_step = 0
GAMMA = 0.99

if modelLoad:
    main_qn.model.load_weights("model.h5")
    adam = Adam(lr=0.001)
    main_qn.model.compile(loss=Huber(), optimizer=adam)
    with open('epsilon.dat', 'rb') as fp:
        epsilon = pickle.load(fp)
    with open('total_step.dat', 'rb') as fp:
        total_step = pickle.load(fp)
    with open('exp.dat', 'rb') as fp:
        memory = pickle.load(fp)

    print("モデル読み込み")

state = env.reset()
state = np.stack((state, state, state, state), axis=2)  # 4フレームを重ねる
state = state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))
multi_step_reward = [0 for _ in range(3)]

success_count = 0
f = False
history = []

while True:
    episode += 1
    step = 0
    if f:
        break
    target_qn.model.set_weights(main_qn.model.get_weights())
    reward_sum = 0
    loss = 0
    action = 0
    for _ in range(1, 2000):
        step += 1
        total_step += 1
        if epsilon > E_STOP and total_step > 3200:
            epsilon -= (E_START - E_STOP) / 3000000

        if step % 2 == 1:
            if epsilon > np.random.rand():
                action = env.action_space.sample()
            else:
                action = np.argmax(main_qn.model.predict(state)[0])

        next_state, reward, done, temp = env.step(action)
        multi_step_reward.append(reward)
        multi_step_reward.pop(0)
        reward_sum += reward
        reward = 0
        for i, r in enumerate(multi_step_reward):
            reward += r * (GAMMA ** i)
        f = temp.pop()
        if f:
            Flappy.pygame.quit()
            break

        next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1], 1)
        s_t1 = np.append(next_state, state[:, :, :, -3:], axis=3)
        memory.add((state, action, reward, s_t1, done))
        if memory.total() >= BATCH_SIZE and total_step > 3200:
            # ニューラルネットワークの入力と出力の準備
            # バッチサイズ分の経験をランダムに取得
            index, minibatch, weight = memory.sample(BATCH_SIZE)

            # ニューラルネットワークの入力と出力の生成
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            # 入力に状態を指定
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = main_qn.model.predict(state_t)
            Q_sa = target_qn.model.predict(state_t1)
            # 採った行動の価値を計算
            output = targets
            err = reward_t + (GAMMA ** 3) * np.max(Q_sa, axis=1) * np.invert(terminal)
            err *= weight
            targets[range(BATCH_SIZE), action_t] = err
            for i in range(BATCH_SIZE):
                td_error_diff = output[i][action_t[i]] - err[i]
                memory.update(index[i], minibatch[i], td_error_diff)
            # 行動価値関数の更新
            loss += main_qn.model.train_on_batch(state_t, targets)

            if episode % 500 == 0:
                target_qn.model.set_weights(main_qn.model.get_weights())
            # print('累計ステップ: {},ACTION: {},epsilon: {:.4f},reward:{:.3f},loss:{:.3f}'.format(total_step, action, epsilon
            #                                                                                , reward,
            #                                                                                loss))
        state = s_t1
        if done:
            break

    # エピソード完了時のログ表示
    print('エピソード: {}, ステップ数: {}, 累計ステップ: {},epsilon: {:.4f},reward:{:.3f},loss:{:.3f}'.format(episode, step, total_step,
                                                                                              epsilon, reward_sum,
                                                                                              loss))
    history.append(step)

    # 環境のリセット
    state = env.reset()
    state = np.stack((state, state, state, state), axis=2)  # 4フレームを重ねる
    state = state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))

num = 20
b = np.ones(num) / num
history = np.convolve(history, b, mode='same')  # 移動平均
plt.plot(history)
plt.show()
main_qn.model.save_weights("model.h5", overwrite=True)
with open('epsilon.dat', 'wb') as fp:
    pickle.dump(epsilon, fp)
with open('total_step.dat', 'wb') as fp:
    pickle.dump(total_step, fp)
with open('exp.dat', 'wb') as fp:
    pickle.dump(memory, fp)
