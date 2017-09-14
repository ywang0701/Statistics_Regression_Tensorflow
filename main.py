# /usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import csv

import tensorflow as tf
import numpy as np
import pandas as pd

def main():
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 7])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    # Y_hist = tf.summary.histogram('Y', Y)

    W = tf.Variable(tf.random_normal([7, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    W_hist  = tf.summary.histogram('W', W)
    b_hist = tf.summary.histogram('b', b)

    # Hypothesis
    hypothesis = tf.matmul(X, W) + b
    # hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    cost_scalr = tf.summary.scalar('cost', cost)

    # Minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    # Launch the graph in a session.
    sess = tf.Session()
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs")
    writer.add_graph(sess.graph)  # Show the graph

    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(100000):
        summary, cost_val, hy_val, W_val, b_val, _ = sess.run(
            [merged_summary, cost, hypothesis, W, b, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            # print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
            print(step, "Cost: ", cost_val)
            print("b: ", b_val)
            print ("W\n", W_val)


    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        W_val = sess.run(W)
        b_val = sess.run(b)
        columns_count = len(W_val)
        header = ['w'+str(i) for i in range(1, columns_count+1)] + ['b']
        writer.writerow(header)
        writer.writerow(list(W_val)+list([b_val]))


    # Ask my score
    # print("Your score will be ", sess.run(
    #     hypothesis, feed_dict={X: [[100, 70, 101]]}))
    #
    # print("Other scores will be ", sess.run(hypothesis,
    #                                         feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

if __name__ == "__main__":
    tf.set_random_seed(777)  # for reproducibility

    try:
        csv_file = sys.argv[2]
    except:
        # sys.exit("Not specified data file")
        csv_file = '00_nom_135_records.csv'

    xy = pd.read_csv(csv_file, delimiter=',', dtype=np.float32).as_matrix(columns=None)

    x_data = xy[:, 0:-1] # values of dependent variables
    y_data = xy[:, [-1]] # values of independent variable

    main()