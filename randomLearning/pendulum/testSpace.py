import tensorflow as tf
import pdb 
# Symbolic Tensor
a = tf.constant(3)
b = tf.constant(4)
c = tf.add(a, b)  # Symbolic tensor representing an addition operation

# Normal Tensor
d = tf.constant([1, 2, 3])  # Normal tensor with concrete numerical values

# Print tensor types
print("Type of 'a':", type(a))  # Output: <class 'tensorflow.python.framework.ops.Tensor'>

print("Type of 'c':", type(c))  # Output: <class 'tensorflow.python.framework.ops.Tensor'>

print("Type of 'd':", type(d))  # Output: <class 'tensorflow.python.framework.ops.Tensor'>

pdb.set_trace() 
# Access numerical values (applicable only for normal tensors)
#with tf.Session() as sess:  # For TensorFlow 1.x
#    print("Value of 'd':", sess.run(d))  # Output: [1 2 3]
