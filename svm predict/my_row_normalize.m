function out = my_row_normalize(X)
[N, ~] = size(X);
max_X = max(X, [], 1);
min_X = min(X, [], 1);
out = (X-ones(N, 1)*min_X)./(ones(N, 1)*(max_X-min_X));