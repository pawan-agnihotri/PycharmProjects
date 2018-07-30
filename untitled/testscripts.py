import tensorflow as tf
# source ops - they dont need any input like consts
hello = tf.constant('Hello, TensorFlow!')
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(5.0, dtype=tf.float32)
# make operation using above variable
total = tf.add(a,b)
multi= total*b
# define variables
state = tf.Variable(0.0)
update = tf.assign(state, multi)
# define placeholder
x=tf.placeholder(tf.float32)
y=x*2
# define dataset
my_data = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
# define layers
x_layer = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y_layer = linear_model(x_layer)

# create a session and run the operations
with tf.Session() as sess:
    print(sess.run(hello))
    print(sess.run(total))
    print(sess.run(multi))
    init = tf.global_variables_initializer()
    print("using loop to change the variable")
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

#  To pass the  data, we call session’s run method with feed_dict
    print("feed data to variable using dictionary")
    print(sess.run(y, feed_dict={x: 3}))
    dictionary = {x: [[1, 2], [3, 4]]}
    print(sess.run(y, feed_dict=dictionary))
    # print data from dataset
    print("dataset")
    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            break
    #print layers
    print("layers")
    sess.run(init)
    dictionary = {x_layer: [[1, 2, 5], [3, 4, 6]]}
    print(sess.run(y_layer, feed_dict=dictionary))

    writer = tf.summary.FileWriter('C:\\tmp')
    writer.add_graph(tf.get_default_graph())
    writer.close()
