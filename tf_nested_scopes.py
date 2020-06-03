#!/usr/bin/env python3

import tensorflow.compat.v1 as tf


"""
Exploring scoping for tf variables and how their initialization is affected

* tf.AUTO_REUSE can be used to share parameters between different variable scopes
* For initialization, it first looks at what is defined in the scope. If it is not 
  defined then it check what is defined in the variable object constructor. If not, 
  it just uses glorot normal.
* The initializer defined in the scope can be overridden by the variable object
  constructor
"""


with tf.variable_scope('scope1', reuse=tf.AUTO_REUSE) as outer_scope:
    v1 = tf.get_variable('v1', shape=[1])
    v2 = tf.get_variable('v2', shape=[1], initializer=tf.random_normal_initializer) 
    v2_2 = tf.get_variable('v2', shape=[1])  # duplicate var names

    print(100 * '-') 
    print(v1.name, v1.initial_value)  # scope1/v1:0
    print(v2.name, v2.initial_value)  # scope1/v2:0
    print(v2_2.name, v2_2.initial_value)  # scope1:v2:0 (require tf.AUTO_REUSE)
    print(100 * '-') 

    # try out nested scopes
    with tf.variable_scope('scope2', initializer=tf.initializers.glorot_normal):
        v3 = tf.get_variable('v3', shape=[1], initializer=tf.random_uniform_initializer)  # override
        v2 = tf.get_variable('v2', shape=[1])
        print(v3.name, v3.initial_value)  # scope1/scope2/v3:0
        print(v2.name, v2.initial_value)  # scope1/scope2/v2:0 (it is allowed since different scopes)

    print(100 * '-')
    
    # it will still use the intializer from scope1 defined first
    with tf.variable_scope(outer_scope, initializer=tf.random_uniform_initializer) as inner_scope:
        v4 = tf.get_variable('v4', shape=[1])
        v2 = tf.get_variable('v2', shape=[1])
        print(v4.name, v2.initial_value)  # scope1/v4:0
        print(v2.name, v2.initial_value)  # scope1/v2:0 (reused)

    print(100 * '-')         
    

