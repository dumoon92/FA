close all
clear

N_min = 5;
N_max = 50;

N_range = N_min:5:N_max;

N_num = length(N_range);
time_svm = zeros(N_num, 1);
time_krig = zeros(N_num, 1);
rmse_svm = zeros(N_num, 1);
rmse_krig = zeros(N_num, 1);

tic
for i = 1:N_num
    N = N_range(i);
    x = linspace(0,2,N)';
    y = humps(x);
    svm_kernel = 'polynomial';
    krig_kernel = 'ardexponential';

    x_test = linspace(0,2,2*N)';
    y_test = humps(x_test);
    
    tic
    [ysvm, svmMdl] = my_fitrsvm(x, y, svm_kernel); 
    time_svm(i, 1) = toc;
    y_predict = predict(svmMdl, x_test);
    rmse_svm(i, 1) = sqrt(immse(y_predict, y_test));

    tic
    [ykrig, krigMdl] = my_fitrkrig(x, y, krig_kernel); 
    time_krig(i, 1) = toc;
    y_predict = predict(krigMdl, x_test);
    rmse_krig(i, 1) = sqrt(immse(y_predict, y_test));
    my_plot(x, y, ysvm, ykrig, N, svm_kernel)   
end
toc

figure
hold on
grid on
plot(N_range, time_krig)
plot(N_range, time_svm)
legend('Krig', 'SVM')
title('Training Time for 2D')
xlabel('Data Amount N')
ylabel('Training Time (Seconds)')
hold off


figure
hold on
grid on
plot(N_range, rmse_krig)
plot(N_range, rmse_svm)
title('RMSE for 2D')
legend('Krig', 'SVM')
xlabel('Data Amount N')
ylabel('RMSE')
hold off