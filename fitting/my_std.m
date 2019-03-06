function output = my_std(input)
mu = mean(input);
sig = std(input);

output = (input - mu) / sig;