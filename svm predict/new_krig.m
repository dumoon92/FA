clear
close  all
clc
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;
y_raw=data.Data;

mesh_dencity = 3;
% data_set_num_set = floor(linspace(1e1, 5e2, mesh_dencity));
% train_len_set = floor(linspace(1e1, 5e2, mesh_dencity)); 

data_set_num_set = [300];
train_len_set = [300]; 

parameter_str = strcat('-',num2str(data_set_num_set(end)), '-', num2str(train_len_set(end)),'_');

predict_len = 200; 
start_train_index = 1; 
start_predict_index = 3e4;

svm_kernel = {'gaussian', 'rbf', 'linear', 'polynomial'};
krig_kernel = {'squaredexponential', 'matern32', 'matern52',...
    'ardsquaredexponential'};
kernel_num = numel(krig_kernel);
error_matrix = ones(mesh_dencity, mesh_dencity, kernel_num);
time_matrix = ones(mesh_dencity, mesh_dencity, 4);
for kernel_num = 1:kernel_num
    for i = 1:numel(data_set_num_set)
        data_set_num = data_set_num_set(i);
        for k = 1:numel(train_len_set)
            train_len = train_len_set(k);
            tic
            [test_y, predict_y, error] = my_new_krig(y_raw, data_set_num, train_len, ...
                predict_len, start_train_index, start_predict_index, krig_kernel(kernel_num));
            error_matrix(k, i, kernel_num) = error;
            time_matrix(k, i, kernel_num) = toc;
        end
    end
end

figure('units','normalized','outerposition',[0 0 1 1])  % output graph as full screen
for kernel_num = 1:kernel_num
    subplot(2, 2, kernel_num);
    if strfind(version, '2015')
        h = heatmap(error_matrix(:, :, kernel_num));
        title(strcat('Krig with kernel', krig_kernel(kernel_num)))
    elseif strfind(version, '2018')
        % only for high version MATLAB
        h = heatmap(data_set_num_set, train_len_set, error_matrix(:, :, kernel_num));  
        h.XLabel = 'data set num set';
        h.YLabel = 'train len set';
        h.Title = strcat('Ralative Errors by Krig with kernel: ', krig_kernel(kernel_num));
    end
end
saveas(gcf, strcat('new_krig_error_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
save(strcat('new_krig', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.mat'))
