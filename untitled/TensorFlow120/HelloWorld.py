import tensorflow as tf

Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])
first_operation = tf.add(Matrix_one, Matrix_two)
second_operation = Matrix_one + Matrix_two
third_operation = tf.matmul(Matrix_one, Matrix_two)
a=tf.placeholder(tf.float32)
b=a*2
with tf.Session() as session:
    result = session.run(Scalar)
    print ("Scalar (1 entry):\n %s \n" % result)
    result = session.run(Vector)
    print ("Vector (3 entries) :\n %s \n" % result)
    result = session.run(Matrix)
    print ("Matrix (3x3 entries):\n %s \n" % result)
    result = session.run(Tensor)
    print ("Tensor (3x3x3 entries) :\n %s \n" % result)
    result = session.run(first_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = session.run(second_operation)
    print("Defined using normal expressions :")
    print(result)
    result = session.run(third_operation)
    print("Defined using normal expressions :")
    print(result)
    result = session.run(b,feed_dict={a:3.5})
    print (result)

# plot a line in
#Y = a X + b
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)
X = np.arange(0.0, 5.0, 0.1)
a=1
b=0
Y= a*X + b
plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
