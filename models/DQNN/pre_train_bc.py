import tensorflow as tf
import glob
from DQNN import build_q_network
from data import DatasetEnv
from configs import N_ACTIONS


model = build_q_network()
model.compile(
optimizer=tf.keras.optimizers.Adam(1e-4),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)


X, Y = [], []
for ep in glob.glob('dataset/episodes/*'):
env = DatasetEnv(ep)
s = env.reset()
done = False
while not done:
    a = env.actions[env.t]
    X.append(s)
    Y.append(a)
    s, _, done = env.step(a)


X = tf.convert_to_tensor(X)
Y = tf.convert_to_tensor(Y)


model.fit(X, Y, batch_size=32, epochs=10)
model.save_weights('bc_weights.h5')