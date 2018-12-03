function x = my_remove_nan(x)
% trick for NaN data
nan_index = find(isnan(x));
for i = 1:length(nan_index)
    x(nan_index(i)) = 0.5;
end