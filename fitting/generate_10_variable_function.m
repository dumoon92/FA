%% Generate W with uniformly distributed random variables
function [X, W] = generate_10_variable_function(N, variable)
% empty variable list means to use all variables, 
% else means to use given variables, the rest variables will be fixed as
% Baseline

if isequal(variable, [])
    variable = ["S_w", "W_fw", "A", "Lambda", "q", "lambda", "tC", "N_L", "W_dg", "W_p"];
end

% for all random number use uniformly distribution

if ismember("S_w", variable)
    S_w_range = [150, 200];
    S_w = my_rand(N, 1, S_w_range, "uniform");
else
    S_w = ones(N, 1)*174;
end

if ismember("W_fw", variable)
    W_fw_range = [220, 300];
    W_fw = my_rand(N, 1, W_fw_range, "uniform");
else
    W_fw = ones(N, 1)*252;
end

if ismember("A", variable)
    A_range = [6, 10];
    A = my_rand(N, 1, A_range, "uniform");
else
    A= ones(N, 1)*7.52;
end

if ismember("Lambda", variable)
    Lambda_range = [-10, 10];
    Lambda = my_rand(N, 1, Lambda_range, "uniform")/180*pi;  % digree to radius
else
    Lambda = ones(N, 1)*0;
end

if ismember("q", variable)
    q_range = [16, 45];
    q = my_rand(N, 1, q_range, "uniform");
else
    q = ones(N, 1)*34;
end

if ismember("lambda", variable)
    lambda_range = [0.5, 1];
    lambda = my_rand(N, 1, lambda_range, "uniform");
else
    lambda = ones(N, 1)*0.672;
end

if ismember("tC", variable)
    tC_range = [0.08, 0.18];
    tC = my_rand(N, 1, tC_range, "uniform");
else
    tC = ones(N, 1)*0.12;
end

if ismember("N_L", variable)
    N_L_range = [2.5, 6];
    N_L = my_rand(N, 1, N_L_range, "uniform");
else
    N_L = ones(N, 1)*3.8;
end

if ismember("W_dg", variable)
    W_dg_range = [1700, 2500];
    W_dg = my_rand(N, 1, W_dg_range, "uniform");
else
    W_dg = ones(N, 1)*2000;
end

if ismember("W_p", variable)
    W_p_range = [0.025, 0.08];
    W_p = my_rand(N, 1, W_p_range, "uniform");
else
    W_p = ones(N, 1)*0.064;
end

W = 0.036.*S_w.^0.758.*W_fw.^0.0035.*(A./(cos(Lambda)).^2).^0.6.*q.^0.006.*lambda.^0.04.*(100*tC./cos(Lambda)).^-0.3.*(N_L.*W_dg).^0.49+S_w.*W_p;
X = [S_w, W_fw, A, Lambda, q, lambda, tC, N_L, W_dg, W_p];
