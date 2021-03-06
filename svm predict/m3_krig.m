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
start_train_index = 666; 
start_predict_index = 3e4;

krig_kernel = {'squaredexponential', 'matern52'};
kernel_num_set = numel(krig_kernel);
error_matrix = ones(mesh_dencity, mesh_dencity, kernel_num_set);
rmse_matrix = ones(mesh_dencity, mesh_dencity, kernel_num_set);
time_matrix = ones(mesh_dencity, mesh_dencity, kernel_num_set);
for kernel_num = 1:kernel_num_set
    for i = 1:numel(data_set_num_set)
        data_set_num = data_set_num_set(i);
        for k = 1:numel(train_len_set)
            disp([kernel_num, i, k]);
            train_len = train_len_set(k);
            tic
            [test_y, predict_y, error, rmse] = my_new_krig(y_raw, data_set_num, train_len, ...
                predict_len, start_train_index, start_predict_index, krig_kernel(kernel_num));
            error_matrix(k, i, kernel_num) = error;
            rmse_matrix(k, i, kernel_num) = rmse;
            time_matrix(k, i, kernel_num) = toc;
        end
    end
end

%% save mat
save(strcat('new_krig_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.mat'))

%% plot error
for kernel_num = 1:kernel_num_set
figure('units','normalized','outerposition',[0 0 .35 .45])  % output graph as full screen
    if strfind(version, '2015')
        h = heatmap(error_matrix(:, :, kernel_num));
        title(strcat('Krig with kernel', krig_kernel(kernel_num)))
    elseif strfind(version, '2018')
        % only for high version MATLAB
        h = heatmap(data_set_num_set, train_len_set, error_matrix(:, :, kernel_num));  
        h.XLabel = 'Data set number set';
        h.YLabel = 'Train length set';
        h.Title = strcat('Ralative errors by kriging with kernel: ', krig_kernel(kernel_num));
    end
    set(gcf, 'Units', 'inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    saveas(gcf, strcat('m3_krig_error_', string(krig_kernel(kernel_num)), regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
end

%% plot rmse
for kernel_num = 1:kernel_num_set
    figure('units','normalized','outerposition',[0 0 .35 .45])  % output graph as full screen
    if strfind(version, '2015')
        h = heatmap(rmse_matrix(:, :, kernel_num));
        title(strcat('Krig with kernel', krig_kernel(kernel_num)))
    elseif strfind(version, '2018')
        % only for high version MATLAB
        h = heatmap(data_set_num_set, train_len_set, rmse_matrix(:, :, kernel_num));  
        h.XLabel = 'Data set number set';
        h.YLabel = 'Train length set';
        h.Title = strcat('RMSE by krig with kernel: ', krig_kernel(kernel_num));
    end

    set(gcf, 'Units', 'inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    saveas(gcf, strcat('m3_krig_rmse_', string(krig_kernel(kernel_num)), regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
end
%% plot time
for kernel_num = 1:kernel_num_set
    figure('units','normalized','outerposition',[0 0 .35 .45])  % output graph as full screen
    if strfind(version, '2015')
        h = heatmap(time_matrix(:, :, kernel_num));
        title(strcat('Krig with kernel', krig_kernel(kernel_num)))
    elseif strfind(version, '2018')
        % only for high version MATLAB
        h = heatmap(data_set_num_set, train_len_set, time_matrix(:, :, kernel_num));  
        h.XLabel = 'Data set number set';
        h.YLabel = 'Train length set';
        h.Title = strcat('Ralative errors by krig with kernel: ', krig_kernel(kernel_num));
    
    end
    set(gcf, 'Units', 'inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    saveas(gcf, strcat('m3_krig_time_', string(krig_kernel(kernel_num)), regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), parameter_str, '.pdf'));
end
close all

