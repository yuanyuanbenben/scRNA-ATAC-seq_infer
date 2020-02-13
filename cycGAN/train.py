# import
from simple_model_class import scRAP
import simulation_data as sim_data

def get_batch(data,batch_size):
    input_queue = tf.train.slice_input_producer([data], num_epochs=1, shuffle=True, capacity=32 )
    batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    return batch


def train(data, m, n, learning_rate, batch_size, lamda):
    data_x = data["x"]
    data_y = data["y"]
    func = scRAP(m, n)

    with tf.variable_scope("placeholder", reuse=tf.AUTO_REUSE):
        #         Z_x = tf.placeholder(tf.float32,[None,m])
        #         Z_y = tf.placeholder(tf.float32,[None,n])
        X = tf.placeholder(tf.float32, [None, m])
        Y = tf.placeholder(tf.float32, [None, n])

    with tf.variable_scope("cycGAN", reuse=tf.AUTO_REUSE):
        false_y = func.G(X, m)
        false_x = func.F(Y, n)
        false_y_x = func.F(false_y, n)
        false_x_y = func.G(false_x, m)
        false_Dy_logit, false_Dy_prob = func.Dy(false_y)
        real_Dy_logit, real_Dy_prob = func.Dy(Y)
        false_Dx_logit, false_Dx_prob = func.Dx(false_x)
        real_Dx_logit, real_Dx_prob = func.Dx(X)

    with tf.variable_scope("gan_loss", reuse=tf.AUTO_REUSE):
        G_loss = lamda * tf.reduce_mean(tf.square(false_y_x - X)) + lamda * tf.reduce_mean(
            tf.square(false_x_y - Y)) + tf.reduce_mean(tf.square(false_Dy_prob - 1))
        F_loss = lamda * tf.reduce_mean(tf.square(false_y_x - X)) + lamda * tf.reduce_mean(
            tf.square(false_x_y - Y)) + tf.reduce_mean(tf.square(false_Dx_prob - 1))
        Dy_loss = tf.reduce_mean(tf.square(real_Dy_prob - 1)) + tf.reduce_mean(tf.square(false_Dy_prob))
        Dx_loss = tf.reduce_mean(tf.square(real_Dx_prob - 1)) + tf.reduce_mean(tf.square(false_Dx_prob))

    with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
        G_step = tf.train.AdamOptimizer(learning_rate).minimize(G_loss)
        F_step = tf.train.AdamOptimizer(learning_rate).minimize(F_loss)
        Dy_step = tf.train.AdamOptimizer(learning_rate).minimize(Dy_loss)
        Dx_step = tf.train.AdamOptimizer(learning_rate).minimize(Dx_loss)

    init_op = tf.global_variables_initializer()

    x_batch = get_batch(data["x"], batch_size)
    y_batch = get_batch(data["y"], batch_size)
    print(1)
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1000):
            data_x, data_y = sess.run([x_batch, y_batch])
            _, gloss = sess.run([G_step, G_loss], feed_dict={X: data_x, Y: data_y})
            _, floss = sess.run([F_step, F_loss], feed_dict={X: data_x, Y: data_y})
            _, dyloss = sess.run([Dy_step, Dy_loss], feed_dict={X: data_x, Y: data_y})
            _, dxloss = sess.run([Dx_step, Dx_loss], feed_dict={X: data_x, Y: data_y})
            if i % 1 == 0:
                print("Epoch:{5},Gloss:{1},Floss:{2},Dyloss:{3},Dxloss:{4}".format(gloss, floss, dyloss, dxloss, i))

data = sim_data.data
m = sim_data.m
n = sim_data.n
train(data,m,n,0.01,20,1)
