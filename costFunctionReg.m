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
n = length(theta);
t = lambda / (2*m);
for k = 1:n
  J += t * theta(k,1)^2;
endfor


grad = X' * (sigmoid(X * theta) - y) / m + lambda * theta / m;
grad(1,1) -=  lambda * theta(1,1) / m;





% =============================================================

end
