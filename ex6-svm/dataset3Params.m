function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% we construct the search space using the suggested values:
search = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
[CC, ss] = meshgrid(search);
C_search = vec(CC);
sigma_search = vec(ss);

l = size(C_search);
error = zeros(l, 1);

% for each (C, sigma) combination:
for i=1:l
    
  % we train the model on the training set:
  model= svmTrain(X, y, C_search(i), @(x1, x2) gaussianKernel(x1, x2, sigma_search(i)));  
  
  % we test the model on the cross-validation set:
  predictions = svmPredict(model, Xval);
  error(i) = mean(double(predictions ~= yval));
  
endfor

% find minimum error:
[err, imin] = min(error);
err
imin
% output optimal hyperparameters:
C = C_search(imin)
sigma = sigma_search(imin)

% =========================================================================

end
