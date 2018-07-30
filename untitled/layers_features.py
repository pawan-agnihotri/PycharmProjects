import tensorflow as tf
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

# creating features
features = {
    'sales': [[5], [10], [8], [9], [2]],
    'department': ['sports', 'sports', 'gardening', 'gardening', 'media']}
department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening','media'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
# create a session and run the operations
with tf.Session() as sess:
    init = tf.global_variables_initializer()
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
    print("feature and layers")
    table_init = tf.tables_initializer()
    sess.run(table_init)
    print(sess.run(inputs))
    writer = tf.summary.FileWriter('C:\\tmp')
    writer.add_graph(tf.get_default_graph())
    writer.close()
