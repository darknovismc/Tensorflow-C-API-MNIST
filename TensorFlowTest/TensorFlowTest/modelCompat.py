import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x = tf.placeholder(tf.float32, shape=[None,28*28], name='input')
y = tf.placeholder(tf.float32, shape=[None,10], name='target')

#h1 = tf.sigmoid(tf.layers.dense(x, 28*28),name='hidden')
#y_ = tf.sigmoid(tf.layers.dense(h1, 10),name='output')
#loss = tf.reduce_mean(tf.square(y_ - y), name='loss')
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

#simpleRnn1 =  tf.keras.layers.GRU(256,return_sequences=True)
#h1 = simpleRnn1(x)
#simpleRnn2 =  tf.keras.layers.GRU(256)
#h2 = simpleRnn2(h1)
#y_ = tf.identity(tf.layers.dense(h2, 10),name='output')
#loss = tf.nn.softmax_cross_entropy_with_logits_v2(y,y_)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

h1 =  tf.nn.relu(tf.layers.dense(x, 512),name='hidden')
h2 =  tf.nn.relu(tf.layers.dense(h1, 256),name='hidden2')
y_ = tf.identity(tf.layers.dense(h2, 10),name='output')
loss = tf.nn.softmax_cross_entropy_with_logits_v2(y,y_)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train_op = optimizer.minimize(loss, name='train')
init = tf.global_variables_initializer()
saver_def = tf.train.Saver().as_saver_def()
path=tf.get_default_graph().as_graph_def().SerializeToString()
with open('graph.pb', 'wb') as f: f.write(path)




from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print(x_train.shape)

tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

for i in range(60000):
        sess.run(train_op, feed_dict={x:x_train[i].reshape(1,28,28),y:y_train[i].reshape(1,10)})

for el in sess.run(y_, feed_dict={x: x_test.reshape(1,28,28), y: y_test.reshape(1,10)}):
        print('    ',el)