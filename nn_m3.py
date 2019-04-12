import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import datetime
import time
import seaborn as sns; sns.set()

# for single test
# train_len_set = np.array([10, 20])
# data_set_set = np.array([10, 20])
# train_start = 0
# test_start = 30000
# predict_num = 150

# for heatmap
mesh_dencity = 10
train_len_set = np.linspace(10, 500, mesh_dencity, dtype=np.int32)
data_set_set = np.linspace(10, 500, mesh_dencity, dtype=np.int32)

data = scipy.io.loadmat('088IRWaSS7_Wi1d89_C4d3_wave.mat').get('WG10_DHI')['Data'][0][0]
# data = np.squeeze(data)
data = np.transpose(data)
print('data.shape = ', data.shape)
assert(data.shape[0] == 1)

def norm(x):
    return (x - x.min()) / (x.max()-x.min())


data = norm(data)
data = np.array(data, dtype=np.float32)


def method_3(train_len=10, data_set=50, predict_num=10, train_start=10000, test_start=20000):
    parameter_str = '-'+str(data_set)+'-'+str(train_len)+'-'+str(predict_num)+ \
                    '-'+str(train_start)+'-'+str(test_start)+'_'

    predict_len = 1

    test_len = train_len
    test_set = 666

    test_x = data[0: 1, test_start-train_len: test_start]
    test_y = data[0, test_start: test_start+predict_num]
    results = np.zeros((predict_num, ))

    lstm_size = 30
    lstm_layers = 2
    batch_size = 64
    for model_index in range(predict_num):
        if model_index%50 == 0:
            print('model_index = ', model_index)
        tf.reset_default_graph()
        train_x = np.zeros((data_set, train_len))
        train_y = np.zeros((data_set,))
        for i, train_start_index in enumerate(range(train_start, train_start + data_set)):
            train_x[i, :] = data[0: 1, train_start_index: train_start_index + train_len]
            train_y[i] = data[0, train_start_index
                                 + train_len + model_index: train_start_index + train_len + predict_len + model_index]

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
        outputs = outputs[:, -1]

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

                    # if iteration % 100 == 0:
                    #     print('Epochs:{}/{}'.format(e, epochs),
                    #           'Iteration:{}'.format(iteration),
                    #           'Train loss: {:.8f}'.format(loss))
                    iteration += 1


        with session.as_default() as sess:
            # method 3: train 100 models
            feed_dict = {x: test_x[:, :, None], keep_prob: 1.0}
            results[model_index] = sess.run(predictions, feed_dict=feed_dict)

    f = plt.figure()
    plt.plot(results, 'r', label='predicted wave')
    plt.plot(test_y, 'g--', label='real wave')
    plt.legend()
    plt.title('Wave elevation of neutral network under different inputs')
    plt.xlabel('Data point index')
    plt.ylabel('Wave elevation')
    plt.grid(b=True, which='minor')
    # plt.show()

    date_str = str(datetime.datetime.now()).replace(' ', '').replace(':', '_').replace('.', '_')
    f.savefig("nn_predict-m3-" + parameter_str + date_str + ".pdf")

    f = plt.figure()
    plt.plot(np.abs(results-test_y))
    plt.legend()
    plt.title('Average relative error of neutral network under different inputs')
    plt.xlabel('Data point index')
    plt.ylabel('Average relative error')
    plt.grid(b=True, which='minor')
    plt.show()

    date_str = str(datetime.datetime.now()).replace(' ', '').replace(':', '_').replace('.', '_')
    f.savefig("nn_predict_error-m3-" + parameter_str + date_str + ".pdf")

    return test_y, results


error_matrix = np.zeros((mesh_dencity, mesh_dencity))
rmse_matrix = np.zeros((mesh_dencity, mesh_dencity))
time_matrix = np.zeros((mesh_dencity, mesh_dencity))

for k, train_len in enumerate(train_len_set):
    for j, data_set in enumerate(data_set_set):
        print('mesh grid = ', (k, j))
        time_start = time.clock()
        test_y, predict_y = method_3(train_len=train_len, data_set=data_set, predict_num=predict_num)
        time_matrix[k, j] = time.clock()-time_start
        error_matrix[k, j] = (np.abs(test_y-predict_y)/test_y).mean()
        rmse_matrix[k, j] = np.sqrt(np.square(np.subtract(test_y, predict_y)).mean())
        print('time, error, rmse = ', (time_matrix[k, j], error_matrix[k, j], rmse_matrix[k, j]))

plt.figure()
ax = sns.heatmap(rmse_matrix, annot=True, fmt=".2f", linewidths=.5, xticklabels=train_len_set, yticklabels=data_set_set)
ax.xaxis.tick_top()
ax.set_xlabel('Data Set')
ax.set_ylabel('Train Length')
fig = ax.get_figure()
fig.savefig('nn_rmse.pdf', dpi=400)

plt.figure()
ax2 = sns.heatmap(error_matrix, annot=True, fmt=".2f", linewidths=.5, xticklabels=train_len_set, yticklabels=data_set_set)
ax2.xaxis.tick_top()
ax2.set_xlabel('Data Set')
ax2.set_ylabel('Train Length')
fig = ax2.get_figure()
fig.savefig('nn_error.pdf', dpi=400)

plt.figure()
ax3 = sns.heatmap(np.int64(time_matrix), annot=True, annot_kws={'size': 10}, fmt="d", linewidths=.5, xticklabels=train_len_set, yticklabels=data_set_set)
ax3.xaxis.tick_top()
ax3.set_xlabel('Data Set')
ax3.set_ylabel('Train Length')
fig = ax3.get_figure()
fig.savefig('nn_time.pdf', dpi=400)
