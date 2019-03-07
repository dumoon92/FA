%% Fitting a 10 variable function with SVM and Kriging 

clear
close all 
% echo on

task = 'kernel-number-time-rmse';  % number, dimension, kernel, kernel-number-time-rmse
N = 20;
considered_variable = {'N_z', 'A', 'tC', 'S_w', 'W_dg', 'W_p', 'lambda',  'Lambda', 'q', 'W_fw'};
svm_kernel = 'gaussian';
krig_kernel = 'squaredexponential';

switch task
    case 'number'
        %% the number of dataset
        N_train = int64(linspace(1000, 1e4, N));  % training number of points from xxx to xxx, devided by N parts
        N_test = int64(linspace(100, 890, N));  % testing number of points from xxx to xxx, devided by N parts

        svm_time_record = zeros(N, 1);
        svm_rmse_record = zeros(N, 1);
        % svm_Mdl = cell(N, 1);  
        krig_time_record = zeros(N, 1);
        krig_rmse_record = zeros(N, 1);
        % krig_Mdl = cell(N, 1);

        for i = 1:N
            i
            [svm_time_record(i), svm_rmse_record(i), ~, krig_time_record(i),...
                krig_rmse_record(i), ~] = my_fit_nd_time(N_train(i),...
                N_test(i), considered_variable, svm_kernel, krig_kernel);
        end

        %% plot
        figure
        hold on
        grid on
        plot(N_train, krig_time_record)
        plot(N_train, svm_time_record)
        legend('Krig', 'SVM')
        title('Training Time for 10D')
        xlabel('Data Amount N')
        ylabel('Training Time (Seconds)')
        hold off


        figure
        hold on
        grid on
        plot(N_train,krig_rmse_record)
        plot(N_train, svm_rmse_record)
        title('Normalized RMSE for 10D')
        legend('Krig', 'SVM')
        xlabel('Data Amount N')
        ylabel('Normalized RMSE')
        hold off
        
    case 'dimension'
        %% the number of dataset
        N_train = 1e3;  % training number of points from xxx to xxx, devided by N parts
        N_test = 2e2;  % testing number of points from xxx to xxx, devided by N parts

        svm_time_record = zeros(N, 1);
        svm_rmse_record = zeros(N, 1);
        % svm_Mdl = cell(N, 1);  
        krig_time_record = zeros(N, 1);
        krig_rmse_record = zeros(N, 1);
        % krig_Mdl = cell(N, 1);

        for i = 1:10
            i
            [svm_time_record(i), svm_rmse_record(i), ~, krig_time_record(i),...
                krig_rmse_record(i), ~] = my_fit_nd_time(N_train, N_test,...
                considered_variable(1:i), svm_kernel, krig_kernel);
        end

        %% plot
        figure
        hold on
        grid on
        plot(1:10, krig_time_record)
        plot(1:10, svm_time_record)
        legend('Krig', 'SVM')
        title('Training Time for different Dimension Number')
        xlabel('Number of Dimension')
        ylabel('Training Time (Seconds)')
        hold off


        figure
        hold on
        grid on
        plot(1:10,krig_rmse_record)
        plot(1:10, svm_rmse_record)
        title('Normalized RMSE for different Dimension Number')
        legend('Krig', 'SVM')
        xlabel('Number of Dimension')
        ylabel('Normalized RMSE')
        hold off
        
    case 'kernel'
        svm_kernel = {'gaussian', 'rbf', 'linear', 'polynomial'};
        krig_kernel = {'squaredexponential', 'matern32' , 'matern52',...
            'ardsquaredexponential' , 'ardmatern32' , 'ardmatern52'};
        N_train = 1e3;  % training number of points from xxx to xxx, devided by N parts
        N_test = 5e2;  % testing number of points from xxx to xxx, devided by N parts
        
        svm_time_record = zeros(length(svm_kernel), 1);
        krig_time_record = zeros(length(krig_kernel), 1);
        svm_rmse_record = zeros(length(svm_kernel), 1);
        krig_rmse_record = zeros(length(krig_kernel), 1);
        
        for i = 1:length(svm_kernel)
            i
            [svm_time_record(i), svm_rmse_record(i), ~, ~,...
                ~, ~] = my_fit_nd_time(N_train, N_test,...
                considered_variable, svm_kernel(i), krig_kernel(1));
        end
        
        for i = 1:length(krig_kernel)
            i
            [~, ~, ~, krig_time_record(i),...
            krig_rmse_record(i), ~] = my_fit_nd_time(N_train, N_test,...
            considered_variable, svm_kernel(1), krig_kernel(i));
        end
        
    case 'kernel-number-time-rmse'
        svm_kernel = {'gaussian', 'rbf', 'linear', 'polynomial'};
        krig_kernel = {'squaredexponential', 'matern32', 'matern52',...
            'ardsquaredexponential' , 'ardmatern32' , 'ardmatern52'};
        dataset_num = logspace(2, 4, N);  % logspace, like 1e2, 1e3, 1e4...
        svm_time_record = zeros(length(svm_kernel), N);
        krig_time_record = zeros(length(krig_kernel), N);
        svm_rmse_record = zeros(length(svm_kernel), N);
        krig_rmse_record = zeros(length(krig_kernel), N);
        for i = 1:length(svm_kernel)
            svm_kernel(i)
            for k = 1:length(dataset_num)
                int64(dataset_num(k))
                [svm_time_record(i, k), svm_rmse_record(i, k), ~, krig_time_record(i, k),...
                krig_rmse_record(i, k), ~] = my_fit_nd_time(dataset_num(k)*0.7,...
                dataset_num(k)*0.3, considered_variable, svm_kernel(i), krig_kernel(i));
            end
        end
        
        %% plot
    switch task
        case 'kernel'
            figure
            hold on
            grid on
            subplot(2, 1, 1)
%             sgtitle('Training Time for different Kernels')
%             bar(categorical(krig_kernel), krig_time_record);
            plot(krig_time_record);
            legend('Krig')
            xlabel('Kernel Names')
            ylabel('Training Time (Seconds)')            
            subplot(2, 1, 2)
%             bar(categorical(svm_kernel), svm_time_record);
            plot(svm_time_record);
            legend('SVM')
            xlabel('Kernel Names')
            ylabel('Training Time (Seconds)')
            hold off


            figure
%             sgtitle('Normalized RMSE for different Kernels')
            hold on
            subplot(2, 1, 1)
            grid on
%             bar(categorical(krig_kernel), krig_rmse_record)
            plot(krig_rmse_record);
            legend('Krig')
            xlabel('Kernel Names')
            ylabel('Normalized RMSE')
            subplot(2, 1, 2)
%             bar(categorical(svm_kernel), svm_rmse_record)
            plot(svm_rmse_record);
            legend('SVM')
            xlabel('Kernel Names')
            ylabel('Normalized RMSE')
            hold off
            
        case 'kernel-number-time-rmse'
            figure
%             hold on
%             grid on
            semilogy(dataset_num, krig_time_record);
            title('Training Time for different Krig Kernels')
            xlabel('Number of Data')
            ylabel('Training Time (Seconds)')
            legend('squaredexponential', 'matern32', 'matern52',...
            'ardsquaredexponential' , 'ardmatern32' , 'ardmatern52');
%             hold off;
            figure
%             hold on;
            grid on;
            semilogy(dataset_num, svm_time_record)
            legend('gaussian', 'rbf', 'linear', 'polynomial')
            title('Training Time for different SVM Kernels')
            xlabel('Number of Data')
            ylabel('Training Time (Seconds)')
%             hold off
            
            figure
%             hold on
            grid on
            semilogy(dataset_num, krig_rmse_record);
            title('RMSE for different Krig Kernels')
            xlabel('Number of Data')
            ylabel('RMSE')
            legend('squaredexponential', 'matern32', 'matern52',...
            'ardsquaredexponential' , 'ardmatern32' , 'ardmatern52');
%             hold off;
            figure
%             hold on;
            grid on;
            semilogy(dataset_num, svm_rmse_record)
            legend('gaussian', 'rbf', 'linear', 'polynomial')
            title('RMSE for different SVM Kernels')
            xlabel('Number of Data')
            ylabel('RMSE')
%             hold off           
        
        otherwise
            figure
            hold on
            grid on
            plot(1:length(krig_kernel), krig_time_record);
            plot(1:length(svm_kernel), svm_time_record)
            legend('Krig', 'SVM')
            title('Training Time for different Dimension Number')
            xlabel('Number of Dimension')
            ylabel('Training Time (Seconds)')
            hold off
            
            figure
            hold on
            grid on
            plot(1:length(krig_kernel), krig_rmse_record)
            plot(1:length(svm_kernel), svm_rmse_record)
            title('Normalized RMSE for different Dimension Number')
            legend('Krig', 'SVM')
            xlabel('Number of Dimension')
            ylabel('Normalized RMSE')
            hold off
        end 


end
      