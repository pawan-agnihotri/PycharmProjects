import tensorflow as tf
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10,6)

#x = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
x = np.random.uniform(0.0, 1.0, size=10)
y = np.random.uniform(0.0, 10.0, size=10)
#pred=mX+c
X = tf.placeholder("float")
Y = tf.placeholder("float")
m = tf.Variable(np.random.randn(), name="weight")
c = tf.Variable(np.random.randn(), name="bias")

pred = tf.add(tf.multiply(X, m), c)
loss = tf.reduce_mean(tf.square(pred - Y))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# create a session and run the operations
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        for (x1, y1) in zip(x,y):
            sess.run(optimizer, feed_dict={X: x1, Y: y1})
    print("Optimization Finished!")

    training_cost = sess.run(optimizer, feed_dict={X: x, Y: y})
    print("W=", sess.run(m), "b=", sess.run(c), '\n')

    # Graphic display
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, sess.run(m) * x + sess.run(c), label='Fitted line')
    plt.legend()
    plt.show()
    # Testing example, as requested (Issue #2)
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    for step in range(1000):
        for (x1, y1) in zip(test_X,test_Y):
            sess.run(optimizer, feed_dict={X: x1, Y: y1})
    print("Optimization Finished!")

    sess.run(optimizer, feed_dict={X: x, Y: y})
    print("W=", sess.run(m), "b=", sess.run(c), '\n')

    # Graphic display
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, sess.run(m) * x + sess.run(c), label='Fitted line')
    plt.legend()
    plt.show()

    writer = tf.summary.FileWriter('C:\\tmp')
    writer.add_graph(tf.get_default_graph())
    writer.close()
