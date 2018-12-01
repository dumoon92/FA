clear
clc
close all 

N = 5;
considered_variable = ["S_w", "W_fw", "A", "Lambda", "q", "lambda", "tC", "N_L", "W_dg", "W_p"];
N_train = int64(linspace(1000, 10000, N));
N_test = int64(linspace(100, 1000, N));

svm_time_record = zeros(N, 1);
svm_rmse_record = zeros(N, 1);
% svm_Mdl = cell(N, 1);  
krig_time_record = zeros(N, 1);
krig_rmse_record = zeros(N, 1);
% krig_Mdl = cell(N, 1);

for i = 1:N
    i
    [svm_time_record(i), svm_rmse_record(i), ~, krig_time_record(i), krig_rmse_record(i), ~] = my_fit_nd_time(N_train(i), N_test(i), considered_variable);
end

%% plot

figure
hold on
grid on
plot(N_train, krig_time_record)
plot(N_train, svm_time_record)
legend('Krig', 'SVM')
title('Training Time for 4D')
xlabel('Data Amount N')
ylabel('Training Time (Seconds)')
hold off


figure
hold on
grid on
plot(N_train,krig_rmse_record)
plot(N_train, svm_rmse_record)
title('RMSE for 4D')
legend('Krig', 'SVM')
xlabel('Data Amount N')
ylabel('RMSE')
hold off