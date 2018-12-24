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

predictions = sigmoid(X*theta);
J = sum(-y .* log(predictions) - log(1 - predictions) + y .* log(1 - predictions)) / m + lambda / (2 * m) * sum( theta(2:end) .^2);

n = size(X, 2);                          % number of variables
errors = (predictions - y);              % errors     m * 1
errors_repeated = repmat(errors, 1, n);  % m * n

grad_without_regularization = sum(X.*errors_repeated) / m;

theta(1) = 0;
regularization_param = lambda / m .* theta;

grad = grad_without_regularization + regularization_param';

% =============================================================

end
