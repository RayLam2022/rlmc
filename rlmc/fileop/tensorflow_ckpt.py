
import tensorflow.compat.v1 as tf1

tf1.disable_v2_behavior()


ckpt_path = r"D:\work\jupyfile\trainmodel\modl-1"
reader = tf1.train.NewCheckpointReader(ckpt_path)

saver = tf1.train.import_meta_graph(ckpt_path + ".meta", clear_devices=True)
graph = tf1.get_default_graph()
with tf1.Session(graph=graph) as sess:
    sess.run(tf1.global_variables_initializer())
    saver.restore(sess, ckpt_path)
    print(sess.graph.collections)
    print(sess.graph.get_collection("variables"))


# 获取 变量名: 形状
vars = reader.get_variable_to_shape_map()
for k in sorted(vars):
    print(k, vars[k])

# 获取 变量名: 类型
vars = reader.get_variable_to_dtype_map()
for k in sorted(vars):
    print(k, vars[k])

# 获取张量的值
value = reader.get_tensor("bias")
print(value)
