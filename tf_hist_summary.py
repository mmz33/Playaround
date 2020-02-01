import tensorflow.compat.v1 as tf

# plotting histograms using Tensorboard

x = tf.constant([[[1,2,3], [4,5,6]], [[10,11,12], [15,16,17]]])
sess = tf.Session()
writer = tf.summary.FileWriter('./logs/test_hist', sess.graph) # create summary file writer
tf.summary.histogram("x_hist", x) # add histogram to summary
merge = tf.summary.merge_all() # create a tensor object for all summaries (maybe?)
summary = sess.run(merge) # pass summary info to run and get the summary data
writer.add_summary(summary) # write the summary data using the file writer
writer.flush() # flush to make sure data is written!
