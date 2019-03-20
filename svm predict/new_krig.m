clear
close  all
clc
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;
y_raw=data.Data;

mesh_dencity = 3;
data_set_num_set = floor(linspace(1e1, 5e2, mesh_dencity));
train_len_set = floor(linspace(1e1, 5e2, mesh_dencity)); 

predict_len = 100; 
start_train_index = 1; 
start_predict_index = 3e4;

svm_kernel = {'gaussian', 'rbf', 'linear', 'polynomial'};
krig_kernel = {'squaredexponential', 'matern32', 'matern52',...
    'ardsquaredexponential'};
kernel_num = numel(krig_kernel);
error_matrix = ones(mesh_dencity, mesh_dencity, kernel_num);
for kernel_num = 1:kernel_num
    
    for i = 1:numel(data_set_num_set)
        data_set_num = data_set_num_set(i);
        for k = 1:numel(train_len_set)
            train_len = train_len_set(k);
            [test_y, predict_y, error] = my_new_krig(y_raw, data_set_num, train_len, ...
                predict_len, start_train_index, start_predict_index, krig_kernel(kernel_num));
            error_matrix(k, i) = error;
        end
    end
end

figure('units','normalized','outerposition',[0 0 1 1])  % output graph as full screen
for kernel_num = 1:kernel_num
    subplot(2, 2, kernel_num);
%     h = heatmap(error_matrix(:, :, kernel_num));
    
    % only for high version MATLAB
    h = heatmap(data_set_num_set, train_len_set, error_matrix(:, :, kernel_num));  
    h.XLabel = 'data set num set';
    h.YLabel = 'train len set';
    h.Title = 'Ralative Errors in different parameters';
    
    title(strcat('Krig with kernel', krig_kernel(kernel_num)))
end

saveas(gcf, strcat('new_krig_error_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
save('new_krig.mat')
