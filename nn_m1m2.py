import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import datetime


# data = scipy.io.loadmat('088IRWaSS7_Wi1d89_C4d3_wave.mat')['WG10_DHI']
data = np.transpose(scipy.io.loadmat('matlab.mat')['data'])


def norm(x):
    return (x - x.min()) / (x.max()-x.min())


data = norm(data)
data = np.array(data, dtype=np.float32)

def method_1_and_2(method, train_len, train_start=0, test_start=20000):

    train_set = 500

    predict_len = 1

    train_x = np.zeros((train_set, train_len))
    train_y = np.zeros((train_set, ))

    for i in range(train_start, train_start+train_set):
        train_x[i, :] = data[0, i: i+train_len]
        train_y[i] = data[0, i+train_len: i+train_len+predict_len]

    test_len = train_len
    test_set = 666

    test_x = np.zeros((test_set, test_len))
    test_y = np.zeros((test_set, ))
    for i in range(test_start-test_len, test_start-test_len+test_set):
        test_x[i-test_start, :] = data[0, i: i+test_len]
        test_y[i-test_start] = data[0, i+test_len: i+test_len+predict_len]

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    lstm_size = 30
    lstm_layers = 2
    batch_size = 64

    x = tf.placeholder(tf.float32, [None, train_len, 1], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    # 有lstm_size个单元
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # 添加dropout
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    # 一层不够，就多来几层
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)
    cell = tf.contrib.rnn.MultiRNNCell([ lstm_cell() for _ in range(lstm_layers)])

    # 进行forward，得到隐层的输出
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    # 在本问题中只关注最后一个时刻的输出结果，该结果为下一个时刻的预测值
    outputs = outputs[:,-1]

    # 定义输出层, 输出值[-1,1]，因此激活函数用tanh
    predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.tanh)
    # 定义损失函数
    cost = tf.losses.mean_squared_error(y_, predictions)
    # 定义优化步骤
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    # 获取一个batch_size大小的数据
    def get_batches(X, y, batch_size=64):
        for i in range(0, len(X), batch_size):
            begin_i = i
            end_i = i + batch_size if (i+batch_size) < len(X) else len(X)

            yield X[begin_i:end_i], y[begin_i:end_i]


    epochs = 20
    session = tf.Session()
    with session.as_default() as sess:
        # 初始化变量
        tf.global_variables_initializer().run()

        iteration = 1
        for e in range(epochs):
            for xs, ys in get_batches(train_x, train_y, batch_size):
                # xs[:,:,None] 增加一个维度，例如[64, 20] ==> [64, 20, 1]，为了对应输入
                # 同理 ys[:,None]
                feed_dict = {x: xs[:, :, None], y_: ys[:, None], keep_prob: .5}

                loss, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

                if iteration % 100 == 0:
                    print('Epochs:{}/{}'.format(e, epochs),
                          'Iteration:{}'.format(iteration),
                          'Train loss: {:.8f}'.format(loss))
                iteration += 1


    with session.as_default() as sess:
        if method==1:
            # method 1: no update
            test_x = data[0:1, test_start:test_start + test_len]
            predict_y = np.zeros((1000, ))
            print(test_x.shape)
            for i in range(1000):
                feed_dict = {x: test_x[:, :, None], keep_prob: 1.0}
                results = sess.run(predictions, feed_dict=feed_dict)
                predict_y[i] = results
                test_x = np.append(test_x[:, :-1], results, axis=1)
            results = predict_y

        elif method==2:
            # method 2: update with every new point
            feed_dict = {x: test_x[:, :, None], keep_prob: 1.0}
            results = sess.run(predictions, feed_dict=feed_dict)

        results = results[:, 0]  # 2D -> 1D
        f = plt.figure()
        plt.plot(results, 'r', label='predicted wave')
        print(test_y.shape)
        plt.plot(test_y, 'g--', label='real wave')
        plt.legend()
        plt.title('NN prediction vs real with train length = ' + str(train_len))
        plt.xlabel('data points')
        plt.ylabel('wave height(normalized)')
        plt.show()

        date_str = str(datetime.datetime.now()).replace(' ', '').replace(':', '_').replace('.', '_')
        f.savefig("nn_predict-" + str(method) + "-" + str(train_len) + '_' + date_str + ".pdf")
        return test_y, results


test_y, predict_y = method_1_and_2(method=2, train_len=100)
print(test_y.shape, predict_y.shape)

