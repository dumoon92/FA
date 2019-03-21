clear
close  all
clc
data=load('088IRWaSS7_Wi1d89_C4d3_wave.mat');
data=data.WG10_DHI;
x_raw=data.Time;
y_raw=data.Data;

mesh_dencity = 10;
data_set_num_set = floor(linspace(1e1, 5e2, mesh_dencity));
train_len_set = floor(linspace(1e1, 5e2, mesh_dencity)); 
parameter_str = strcat('-',num2str(data_set_num_set(end)), '-', num2str(train_len_set(end)),'_');

predict_len = 100; 
start_train_index = 1; 
start_predict_index = 3e4;

svm_kernel = {'gaussian', 'rbf', 'linear', 'polynomial'};
krig_kernel = {'squaredexponential', 'matern32', 'matern52',...
    'ardsquaredexponential'};
error_matrix = ones(mesh_dencity, mesh_dencity, 4);
for kernel_num = 1:4
    for i = 1:numel(data_set_num_set)
        data_set_num = data_set_num_set(i);
        for k = 1:numel(train_len_set)
            train_len = train_len_set(k);
            [test_y, predict_y, error] = my_new_svm(y_raw, data_set_num, train_len, ...
                predict_len, start_train_index, start_predict_index, svm_kernel(kernel_num));
            error_matrix(k, i, kernel_num) = error;
        end
    end
end

%% plot
figure('units','normalized','outerposition',[0 0 1 1])  % output graph as full screen
for kernel_num = 1:kernel_num
    subplot(2, 2, kernel_num);
    if strfind(version, '2015')
        h = heatmap(error_matrix(:, :, kernel_num));
        title(strcat('SVM with kernel:', svm_kernel(kernel_num)))
    elseif strfind(version, '2018')
        % only for high version MATLAB
        h = heatmap(data_set_num_set, train_len_set, error_matrix(:, :, kernel_num));  
        h.XLabel = 'Data set num set';
        h.YLabel = 'Train len set';
        h.Title = strcat('Ralative Errors by SVM with kernel: ', svm_kernel(kernel_num));
    end
end
saveas(gcf, strcat('new_svm_error_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
save(strcat('new_svm', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.mat'))
