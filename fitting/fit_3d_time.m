close all
clear

kernel = 'gaussian';

N_min = 5;
N_max = 20;
N_range = N_min:1:N_max;

time_svm = zeros(length(N_range), 1);
time_krig = zeros(length(N_range), 1);

rmse_svm = zeros(length(N_range), 1);
rmse_krig = zeros(length(N_range), 1);

for i = 1:length(N_range)
    N = N_range(i);
    
    x = linspace(-2, 2, N);
    y = linspace(-2, 2, N);
    z = cos(x)'*sin(y);
    
    mesh(x,y,z)
    xlabel('x axis'); ylabel('y axis'); zlabel(' axis');
    title('surface z = cos(x)sin(y)');

    [x_vec, y_vec] = meshgrid(x', y');
    x_length = length(x);
    y_length = length(y);
    x_vec = reshape(x_vec, [], 1);
    y_vec = reshape(y_vec, [], 1);
    z_vec = reshape(z, [], 1);
    
    x_test = linspace(-2, 2, 2*N);
    y_test = linspace(-2, 2, 2*N);
    z_test = cos(x_test)'*sin(y_test);    
    [x_vec_test, y_vec_test] = meshgrid(x_test', y_test');
    x_vec_test = reshape(x_vec_test, [], 1);
    y_vec_test = reshape(y_vec_test, [], 1);
    z_vec_test = reshape(z_test, [], 1);
    
    tic
    [z_svm, svmMdl] = my_fitrsvm([x_vec, y_vec], z_vec, kernel);
    time_svm(i, 1) = toc;
    z_svm = reshape(z_svm, x_length, y_length);
    z_svm_predict = predict(svmMdl, [x_vec_test, y_vec_test]);
    rmse_svm(i, 1) = sqrt(immse(z_vec_test, z_svm_predict));

    tic
    [z_krig, krigMdl] = my_fitrkrig([x_vec, y_vec], z_vec);
    time_krig(i, 1) = toc;
    z_krig = reshape(z_krig, x_length, y_length);
    z_krig_predict = predict(krigMdl, [x_vec_test, y_vec_test]);
    rmse_krig(i, 1) = sqrt(immse(z_vec_test, z_krig_predict));

    figure
    mesh(x,y,z_svm)
    xlabel('x axis'); ylabel('y axis'); zlabel(' axis');
    title('SVM fitting surface z = cos(x)sin(y)');

    figure
    mesh(x,y,z_krig)
    xlabel('x axis'); ylabel('y axis'); zlabel(' axis');
    title('Krig fitting surface z = cos(x)sin(y)');
end


figure
hold on
grid on
plot(N_range, time_krig)
plot(N_range, time_svm)
legend('Krig', 'SVM')
title('Training Time for 3D')
xlabel('Data Amount N')
ylabel('Training Time (Seconds)')
hold off


figure
hold on
grid on
plot(N_range, mag2db(rmse_krig))
plot(N_range, mag2db(rmse_svm))
title('RMSE for 3D')
legend('Krig', 'SVM')
xlabel('Data Amount N')
ylabel('RMSE (dB)')
hold off
