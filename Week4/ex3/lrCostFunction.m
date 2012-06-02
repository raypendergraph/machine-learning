function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
%           grad = ( 1 / m ) * (sigmoid( X * theta ) - y )' * X;
    newTheta = theta; 
    newTheta(1) = 0;   % because we don't add anything for j = 0;
    
	term1 =  y' * log( sigmoid( X * newTheta ));
	term2 = ( 1 - y )' * log( 1 - ( sigmoid( X * newTheta )));
	lambdaCostTerm = ( lambda / ( 2 * m ) ) .* sum( newTheta .^ 2 )' ;
	J = ( ( -1 / m ) * ( term1 + term2 )) + lambdaCostTerm;

	%grad(1)     = ( 1/m ) *  [sigmoid( X * newTheta) - y]' * X(:,1);
	grad = ( 1/m ) * [[sigmoid( X * newTheta) - y]' * X] \
                  + ((lambda / m) * newTheta');
size(grad)










% =============================================================

grad = grad(:);

end
