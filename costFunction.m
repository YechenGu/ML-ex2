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
% Note: grad should have the same as theta
%

h1 = sigmoid(X * theta);
h2 = 1 - sigmoid(X * theta);
[a , b] = size(h1);
for i = 1:a
  for j = 1:b
    h1(i,j) = log(h1(i,j));
    h2(i,j) = log(h2(i,j));
  endfor
endfor
part1 = - y' * h1;
part2 = (1- y)' * h2;
J = (part1 - part2) / m;

grad = X' * (sigmoid(X * theta) - y) / m;




% =============================================================

end
