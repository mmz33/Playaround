import tensorflow as tf

####################################################

# trying different types of scoping such as:
# - tf.variable_score
# - tf.name_scope

def scoping(fn):
    with fn("var_scope"):
        v1 = tf.Variable(1, name="v1", dtype=tf.int32)
        v2 = tf.get_variable("v2", [10], dtype=tf.int32)
        res = tf.add(v1, v2) # op
    print('\n'.join([v1.name, v2.name, res.name]), '\n')
    return res

# var_scope/v1:0
# var_scope/v2:0
# var_scope/Add:0
d1 = scoping(tf.variable_scope)

# var_scope_1/v1:0
# v2:0
# var_scope_1/Add:0
d2 = scoping(tf.name_scope)

####################################################

# trying out reusing parameters with different name scopes
# by using tf.get_variable and tf.variable_score with reuse set to True

with tf.name_scope("name_scope1"):
    with tf.variable_scope("vars"):
        var1 = tf.get_variable("v", [1])
with tf.name_scope("name_scope2"):
    with tf.variable_scope("vars", reuse=True):
        var2 = tf.get_variable("v", [1])

assert var1 == var2
print(var1.name) # vars/v:0
print(var2.name) # vars/v:0
