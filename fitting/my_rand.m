%% generate random vector in given range
function out = my_rand(row_num, column_num, rand_range, distribution)

assert(length(rand_range) == 2);
assert(ischar(distribution));

max_rand = max(rand_range);
min_rand = min(rand_range);

if strcmp(distribution, 'normal')
%     out = (max_rand-min_rand).*randn(row_num, column_num) + min_rand;
elseif strcmp(distribution, 'uniform')
    out = (max_rand-min_rand).*rand(row_num, column_num) + min_rand;
end