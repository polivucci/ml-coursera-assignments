function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% define fwd propagation across 1 layer
function a_out = fwd_propagate(theta, a_in)
  a_in = [ones(size(a_in, 1), 1), a_in];
  a_out = sigmoid(a_in * theta');
endfunction

% store the parameters in a cell array for easy access
ThetaJ={Theta1, Theta2};

% propagate through layers
a = X;
for layer=1:2
  a = fwd_propagate(ThetaJ{layer}, a);
end

% predict the most probable class:
[prob, p] = max(a, [], 2);

% =========================================================================


end
