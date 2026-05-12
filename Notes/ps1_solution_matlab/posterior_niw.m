function [B_hat, xpx_Omegainv, IW_param] = posterior_niw(Y, p, Psi, b, Omega)

% Computes formula components in posterior for VAR(p) model
% given data Y and normal-inverse-Wishart prior

% See notation and formulas in Giannone, Lenza & Primiceri (REStat 2015)
% and their online appendix

% Dimensions
[T,n] = size(Y);
k = n*p+1;

% Data matrices
y = Y(p+1:end,:);
Y_lag = lagmatrix(Y,1:p);
x = [ones(T-p,1) Y_lag(p+1:end,:)];

% Auxiliary quantities
b_reshape = reshape(b,k,n);
B_hat = (x'*x + inv(Omega))\(x'*y + Omega\b_reshape);
epsilon_hat = y-x*B_hat;

% Quantities of interest
xpx_Omegainv = x'*x + inv(Omega);
IW_param = Psi + epsilon_hat'*epsilon_hat + (B_hat-b_reshape)'*(Omega\(B_hat-b_reshape));

end