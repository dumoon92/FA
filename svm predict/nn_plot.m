load('NN_heatmap.mat');
data_set_num_set = round(linspace(10, 500, 10));
train_len_set = round(linspace(10, 500, 10));
%% plot rmse
figure('units','normalized','outerposition',[0 0 0.35 0.45])  % output graph as full screen
if strfind(version, '2015')
    h = heatmap(rmse_matrix(:, :));
    title(strcat('SVM with kernel', svm_kernel(kernel_num)))
elseif strfind(version, '2018')
    % only for high version MATLAB
    h = heatmap(data_set_num_set, train_len_set, rmse_NN(:, :));  
    h.XLabel = 'Data set number set';
    h.YLabel = 'Train length set';
    h.Title = strcat('RMSE by NN with different parameters');
end
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m3_nn_rmse_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close
%% plot time
figure('units','normalized','outerposition',[0 0 .35 .45])  % output graph as full screen
if strfind(version, '2015')
    h = heatmap(time_matrix(:, :));
    title(strcat('SVM with kernel', svm_kernel(kernel_num)))
elseif strfind(version, '2018')
    % only for high version MATLAB
    h = heatmap(data_set_num_set, train_len_set, time_NN(:, :));  
    h.XLabel = 'Data set number set';
    h.YLabel = 'Train length set';
    h.Title = strcat('Training time by NN with different parameters');
end
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, strcat('m3_nn_time_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.pdf'));
close
save(strcat('m3_nn_time_', regexprep(datestr(datetime('now')), {'[%() :]+', '_+$'}, {'_', ''}), '.mat'))
