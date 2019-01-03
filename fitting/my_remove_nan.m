function [X, Y] = my_remove_nan(X, Y)
X = X(~isnan(Y), :);
Y = Y(~isnan(Y));
