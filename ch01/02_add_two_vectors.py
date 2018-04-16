import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  ## To deactivate SSE Warnings

v_1 = tf.constant([1,2,3,4], name='v_1')
v_2 = tf.constant([2,1,5,3], name='v_2')
v_add = tf.add(v_1,v_2)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('log02', sess.graph)
    print(sess.run([v_1, v_2,v_add]))

writer.close()

################################################################################
## InteractiveSession makes itself the default session so that
## you can directly call run the tensor Object using eval() without explicitly calling the session

# sess = tf.InteractiveSession()

# v_1 = tf.constant([1,2,3,4])
# v_2 = tf.constant([2,1,5,3])
# I_matrix = tf.eye(5)
# v_add = v_1 + v_2 #tf.add(v_1,v_2)

# print(v_add.eval())
# print(I_matrix.eval())

# sess.close()
