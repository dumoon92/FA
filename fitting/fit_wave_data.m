load('088IRWaSS7_Wi1d89_C4d3_wave.mat', 'WG5_DHI');
data_set = WG5_DHI;

data_set.Data = my_std(data_set.Data);
data_set.Time = my_std(data_set.Time);

test_x = data_set.Time(int64(0.95*8e4):end);
task = 'LSTM';
kernel = '';
normalization = false;
[tic_toc, predict_data] = my_fit_wave_data(data_set, test_x, task, kernel, normalization);
plot(test_x, data_set.Data(int64(0.95*8e4):end), test_x, predict_data);