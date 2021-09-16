function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% compute hypothesis at each training x:
% outputs: vector mx1
h = sigmoid(X*theta);

% compute cost for each example:
% outputs: vector mx1
vecJ = y.*log(h) + (1 .-y).*log(1 .- h);

% compute ridge regularization:
% outputs: vector mx1
vecReg = theta.^2;
vecReg(1,1) = 0.0; % theta_0 must not be regularised
reg = lambda*sum(vecReg)/(2*m);

% compute regularised total cost across all training set:
% outputs: scalar
J = -sum(vecJ)/m + reg;

% compute parameter-wise cost gradient:
% outputs: vector nx1
gradReg = lambda.*theta./m;
gradReg(1,1) = 0.0; % excluding theta_0
grad = X'*(h .-y)./m + gradReg;

% regularize:


% =============================================================

end
