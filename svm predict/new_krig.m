clear
close  all
clc
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;
y_raw=data.Data;

mesh_dencity = 2;
data_set_num_set = floor(linspace(1e1, 5e2, mesh_dencity));
train_len_set = floor(linspace(1e1, 5e2, mesh_dencity)); 

predict_len = 100; 
start_train_index = 1; 
start_predict_index = 3e4;

error_matrix = ones(mesh_dencity, mesh_dencity);
for i = 1:numel(data_set_num_set)
    data_set_num = data_set_num_set(i);
    for k = 1:numel(train_len_set)
        train_len = train_len_set(k);
        [test_y, predict_y, error] = my_new_krig(y_raw, data_set_num, train_len, ...
            predict_len, start_train_index, start_predict_index, 'exponential');
        error_matrix(i, k) = error;
    end
end
figure
h = heatmap(error_matrix);

% only for high version MATLAB
% h = heatmap(data_set_num_set, train_len_set, error_matrix);  
% h.XLabel = 'data set num set';
% h.YLabel = 'train len set';
% h.Title = 'Ralative Errors in different parameters'

saveas(gcf, strcat('new_krig_error_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
save('new_svm.mat')
