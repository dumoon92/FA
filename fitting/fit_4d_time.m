close all
clear
kernel = 'gaussian';

N_min = 5;
N_max = 11;
N_range = N_min:1:N_max;

time_svm = zeros(length(N_range), 1);
time_krig = zeros(length(N_range), 1);

rmse_svm = zeros(length(N_range), 1);
rmse_krig = zeros(length(N_range), 1);

tic
for i = 1:length(N_range)
    N = N_range(i);
    %% train data
    x1 = linspace(-2, 2, N); 
    x2 = linspace(-2, 2, N);
    x3 = linspace(-2, 2, N);

    [x1_vec, x2_vec, x3_vec] = meshgrid(x1', x2', x3');
    x1_vec = reshape(x1_vec, [], 1);
    x2_vec = reshape(x2_vec, [], 1);
    x3_vec = reshape(x3_vec, [], 1);

    z_vec = cos(x1_vec).*sin(x2_vec).*exp(x3_vec); %     function

    train = [x1_vec, x2_vec, x3_vec, z_vec];
    
    
    %% test data
    x1 = linspace(-2, 2, 2*N); 
    x2 = linspace(-2, 2, 2*N);
    x3 = linspace(-2, 2, 2*N);

    [x1_vec, x2_vec, x3_vec] = meshgrid(x1', x2', x3');
    x1_vec = reshape(x1_vec, [], 1);
    x2_vec = reshape(x2_vec, [], 1);
    x3_vec = reshape(x3_vec, [], 1);

    z_vec = cos(x1_vec).*sin(x2_vec).*exp(x3_vec); %     function

    test = [x1_vec, x2_vec, x3_vec, z_vec];
    
    
    %% training
    
%     test data
    X = train(:,1:3);
    z_vec = train(:, 4);  % ideal data
    
%     train data
    X_test = test(:,1:3);
    z_vec_test = test(:, 4);  % ideal data

    tic
    [z_svm, svmMdl] = my_fitrsvm(X, z_vec, kernel);  % training model, z_svm is results, svmMdl is model
    time_svm(i) = toc;  % training time
    z_svm_predict = predict(svmMdl, X_test);  % using model generate fitted data
    rmse_svm(i) = sqrt(immse(z_vec_test, z_svm_predict));   % compare generated fitted data and ideal data

    tic
    [z_krig, krigMdl] = my_fitrkrig(X, z_vec);
    time_krig(i) = toc;
    z_krig_predict = predict(krigMdl, X_test);
    rmse_krig(i) = sqrt(immse(z_vec_test, z_krig_predict));
end
toc
%% plot

figure
hold on
grid on
plot(N_range, time_krig)
plot(N_range, time_svm)
legend('Krig', 'SVM')
title('Training Time for 4D')
xlabel('Data Amount N')
ylabel('Training Time (Seconds)')
hold off


figure
hold on
grid on
plot(N_range, mag2db(rmse_krig))
plot(N_range, mag2db(rmse_svm))
title('RMSE for 4D')
legend('Krig', 'SVM')
xlabel('Data Amount N')
ylabel('RMSE (dB)')
hold off