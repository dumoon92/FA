import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import datetime
import pickle


data = scipy.io.loadmat('088IRWaSS7_Wi1d89_C4d3_wave.mat').get('WG10_DHI')['Data'][0][0]
time = scipy.io.loadmat('088IRWaSS7_Wi1d89_C4d3_wave.mat').get('WG10_DHI')['Time'][0][0]
data = np.squeeze(data)
time = np.squeeze(time)

print('data size =', data.shape)

def norm(x):
    return (x - x.min()) / (x.max()-x.min())


data = norm(data)
data = np.array(data, dtype=np.float32)

def method_1_and_2(method, train_len=100, train_set=500, train_start=1000, test_set=666, test_start=80000):
    assert train_start > train_set
    

    predict_len = 1  # NN's output should be changed if predict_len != 1

    train_x = np.zeros((train_set, train_len))
    train_y = np.zeros((train_set, ))

#   generate training set input
    for i in range(train_start, train_start+train_set):
        temp_index = i-train_start
        train_x[temp_index, :] = data[temp_index: temp_index+train_len]
        train_y[temp_index] = data[temp_index+train_len: temp_index+train_len+predict_len]

    test_len = train_len

    test_x = np.zeros((test_set, test_len))
    test_y = np.zeros((test_set, ))
    for i in range(test_start, test_start+test_set):
        temp_index = i-test_start
        test_x[temp_index, :] = data[temp_index: temp_index+test_len]
        test_y[temp_index] = data[temp_index+test_len:temp_index+test_len+predict_len]

        # test_x[i-test_start, :] = data[0, i: i+test_len]
        # test_y[i-test_start] = data[0, i+test_len: i+test_len+predict_len]

    print('Training set size of x, y:', train_x.shape, train_y.shape, '\nTraining set size of x, y:', test_x.shape, test_y.shape)

    lstm_size = 30
    lstm_layers = 2
    batch_size = 64

    x = tf.placeholder(tf.float32, [None, train_len, 1], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    # lstm_size is number of units
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # add dropout
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    # more layer
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)
    cell = tf.contrib.rnn.MultiRNNCell([ lstm_cell() for _ in range(lstm_layers)])

    # forward, get hidden layers output
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    # only one output, which is the next moment output
    outputs = outputs[:,-1]

    # define output layer, use tanh as activation function
    predictions = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=tf.tanh)
    # define loss function
    cost = tf.losses.mean_squared_error(y_, predictions)
    # define optimizer method
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    # get batch data with batch_size
    def get_batches(X, y, batch_size=64):
        for i in range(0, len(X), batch_size):
            begin_i = i
            end_i = i + batch_size if (i+batch_size) < len(X) else len(X)

            yield X[begin_i:end_i], y[begin_i:end_i]


    # training 
    epochs = 20
    session = tf.Session()
    with session.as_default() as sess:
        # init
        tf.global_variables_initializer().run()

        iteration = 1
        for e in range(epochs):
            for xs, ys in get_batches(train_x, train_y, batch_size):
                feed_dict = {x: xs[:, :, None], y_: ys[:, None], keep_prob: .5}
                loss, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                if iteration % 100 == 0:
                    print('Epochs:{}/{}'.format(e, epochs),
                          'Iteration:{}'.format(iteration),
                          'Train loss: {:.8f}'.format(loss))
                iteration += 1

    # testing
    with session.as_default() as sess:
        if method==1:
            # method 1: no update,
            test_x = data[None, test_start: test_start + test_len]

            print('test_x.shape', test_x.shape)
            predict_y = np.zeros((test_set, ))
            for i in range(test_set):
                feed_dict = {x: test_x[:, :, None], keep_prob: 1.0}
                results = sess.run(predictions, feed_dict=feed_dict)
                predict_y[i] = results
                test_x = np.append(test_x[:, 1:], results, axis=1)
            results = predict_y
            
            pre_train_y = np.zeros((train_set, ))
            train_m1_x = data[None, train_start: train_start + train_len]
            print(train_m1_x.shape)
            for i in range(train_set):
                feed_dict = {x: train_m1_x[:, :, None], keep_prob: 1.0}
                temp_result = sess.run(predictions, feed_dict=feed_dict)
                pre_train_y[i] = temp_result[0, 0]
                train_m1_x = np.append(train_m1_x[0:1, 1:], temp_result, axis=1)


        elif method==2:
            print('train_x', train_x.shape, 'test_x', test_x.shape)
            feed_dict = {x: train_x[:, :, None], keep_prob: 1.0}
            pre_train_y = sess.run(predictions, feed_dict=feed_dict)
            # method 2: update with every new point
            feed_dict = {x: test_x[:, :, None], keep_prob: 1.0}
            results = sess.run(predictions, feed_dict=feed_dict)
            results = results[:, 0]  # 2D -> 1D

    print('results.shape', results.shape)
    f = plt.figure()
    if method == 1:
        # plot training part result and testing part result
        test_plot_start_index = len(pre_train_y)
        plt.plot(np.arange(test_plot_start_index, test_plot_start_index+len(results), 1), results, 'b', label='predicted test data')
        plt.plot(np.arange(test_plot_start_index, test_plot_start_index+len(results), 1), test_y, 'r--', label='real test data')
        plt.plot(np.arange(0, len(pre_train_y), 1), pre_train_y, 'g--', label='predicted training data')
        plt.plot(np.arange(0, len(train_y), 1), train_y, 'k', label='real training data') 
    elif method == 2:
        # plot only testing part result
        plt.plot(np.arange(0, len(results), 1), results, 'r--', label='predicted test data')
        plt.plot(np.arange(0, len(results), 1), test_y, 'b', label='real test data')
 
    plt.legend()
    plt.title('Wave prediction vs real in NN with train length = ' + str(train_len))
    plt.xlabel('Data Point Index')
    plt.ylabel('Wave Elevation')
    plt.show()

    date_str = str(datetime.datetime.now()).replace(' ', '').replace(':', '_').replace('.', '_')
    f.savefig('m'+ str(method) + "_nn" + "_trainlen-" + str(train_len) + '_' + date_str + ".pdf")

    # save variables
    with open('m'+str(method)+'_nn_'+date_str+'.pickle', 'wb') as handle:
        pickle.dump([test_x, test_y, results], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return test_y, results


test_y, predict_y = method_1_and_2(method=2, train_len=100)

