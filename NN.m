clear ; close all; clc

load('ex4data1.mat') ; % X y  loaded


options = optimset('Maxiter', 500);
lambda = 1;

input_layer_size = 400;
hidden_layer_size = 25;


[m num_labels] = size(y);
%num_labels = size(y, 2);
num_labels = 10;
m = size(X, 1);

pred = zeros(m, 1);
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [Theta1(:); Theta2(:)];

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size+1)), ...
hidden_layer_size,input_layer_size+1);
Theta2 = reshape(nn_params(hidden_layer_size * (input_layer_size+1) + 1:end),...
output_layer_size, hidden_layer_size+1);

fprintf("parameters optimized");

fprintf('\nBegin to predict training set.\n');
a1 = [ones(1, m) ; X'];
a2 = [ ones(1,m) ; Theta1 * sigmoid( a1 ) ];
a3 = sigmoid( a2 );
[ans, p] = max(a3);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred ==y)) *100);



