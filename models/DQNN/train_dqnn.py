import tensorflow as tf
import numpy as np
import glob
from DQNN import build_q_network
from DQNN import ReplayBuffer
from data import DatasetEnv
from configs import *


q_net = build_q_network()
target_net = build_q_network()
q_net.load_weights('bc_weights.h5')
target_net.set_weights(q_net.get_weights())


optimizer = tf.keras.optimizers.Adam(LR)
buffer = ReplayBuffer(REPLAY_SIZE)


epsilon = EPS_START
step_count = 0


for ep_dir in glob.glob('data/episodes/*'):
    env = DatasetEnv(ep_dir)
    s = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            a = np.random.randint(N_ACTIONS)
        else:
            q = q_net(s[None], training=False)
            a = int(tf.argmax(q[0]))

        s2, r, done = env.step(a)
        buffer.add(s, a, r, s2, done)
        s = s2

        epsilon = max(EPS_END, epsilon - (EPS_START - EPS_END)/EPS_DECAY_STEPS)
        if len(buffer) > MIN_REPLAY:
            S, A, R, S2, D = buffer.sample(BATCH_SIZE)
            q_next = tf.reduce_max(target_net(S2), axis=1)
            target = R + GAMMA * q_next * (1 - D)

            with tf.GradientTape() as tape:
                q_pred = tf.reduce_sum(q_net(S) * tf.one_hot(A, N_ACTIONS), axis=1)
                loss = tf.reduce_mean((target - q_pred) ** 2)

            grads = tape.gradient(loss, q_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    
    
        if step_count % TARGET_UPDATE == 0:
            target_net.set_weights(q_net.get_weights())
            
        step_count += 1
q_net.save_weights('dqn_weights.h5')