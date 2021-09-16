function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

error_ij = R.*(X*Theta' - Y);   % prediction error for each movie-user combination

J = 0.5*sum(error_ij(:).^2);  % total cost

regularization = 0.5*lambda*( sum(Theta(:).^2) + sum(X(:).^2) ); % regularization term
J += regularization;            % regularized cost

for i=1:num_movies 
  X_grad(i, :) = error_ij(i, :) * Theta; % gradient of J wrt to X for movie i
endfor
X_grad += lambda*X; % regularization term

for j=1:num_users 
  Theta_grad(j, :) = error_ij(:, j)' * X; % gradient of J wrt to Theta for user j
endfor
Theta_grad += lambda*Theta; % regularization term

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
