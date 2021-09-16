function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% compute hypothesis at each training x:
% outputs: vector mx1
h = sigmoid(X*theta);

% compute cost for each single example:
% outputs: vector mx1
vecJ = y.*log(h) + (1 .-y).*log(1 .- h);
% compute (normalised) total cost across all training set:
% outputs: scalar
J = -sum(vecJ)/m;

% compute parameter-wise cost gradient:
% outputs: vector nx1
grad = X'*(h .-y)./m;


% =============================================================

end
