import tensorflow as tf

# "placeholder" for feeding through feed_dict argument in the run() or eval() function call
x = tf.placeholder("float")
y = 2 * x

data = tf.random_uniform([4,5],10)

## All constants, variables, and placeholders will be defined in the computation graph section of the code.
## If we use the print statement in the definition section,
## we will only get information about the type of tensor, and not its value.
# print(x)
# print(y)
# print(data)

with tf.Session() as sess:
    x_data = sess.run(data)
    print(x_data)

    # print(sess.run(y, feed_dict = {x:x_data}))
    y_data = sess.run(y, feed_dict={x:x_data})
    print(y_data)
