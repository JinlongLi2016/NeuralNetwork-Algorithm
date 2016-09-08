function [J grad] = nnCostFunction(nn_params, ...
                                 input_layer_size, ...
                                 hidden_layer_size, ...
                                 num_labels, ...
                                 X, y, lambda)

m = size(X, 1);
                                 
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size+1)), ...
hidden_layer_size, input_layer_size+1);
Theta2 = reshape(nn_params(hidden_layer_size * (input_layer_size+1) + 1:end),...
num_labels, hidden_layer_size+1);

%J = 0;
%Theta1_grad ;
%Theta2_grad;

a1 = [ones(1,m); X'];
a2 = [ones(1,m); sigmoid(Theta1 * a1)];

a3 = sigmoid(Theta2 * a2); % OutPut layer activation

h_theta = a3;

ix = y;
y = zeros(m, num_labels);
for cac = 1:m
  y(cac, ix(cac)) = 1;
end

J = sum( (y' .* log(a3) + (1-y') .* log(1-a3))(:) )/(-m) + ...
       ( sum( ( Theta1(:, 2:end).*Theta1(:, 2:end) )(:)) + ...
         sum( ( Theta2(:, 2:end).*Theta2(:, 2:end) )(:)) )*lambda/(2*m);


delta3 = h_theta - y';  % 10*m Output layer:

delta2 = (Theta2' * delta3) .* (a2 .* (1-a2));
delta1 = (Theta1' * delta2(2:end, :)) .* (a1 .* (1-a1));


DELTA1 = delta2(2:end, :) * a1';
DELTA2 = delta3*a2'; % OUTPUT LAYER 3. Therefore, no bia unit

Theta1_grad = DELTA1/m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1(:, 2:end)*lambda/m;
Theta2_grad = DELTA2/m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2(:, 2:end)*lambda/m;

grad = [Theta1_grad(:); Theta2_grad(:)];

end


