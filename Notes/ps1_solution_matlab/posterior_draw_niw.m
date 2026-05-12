function [B_draw, Sigma_draw] = posterior_draw_niw(Y, p, Psi, d, b, Omega)

% Generates posterior parameter draw for VAR(p) model
% given data Y and normal-inverse-Wishart prior

% See notation and formulas in Giannone, Lenza & Primiceri (REStat 2015)
% and their online appendix


% Dimensions
[T,n] = size(Y);

% Formula components
[B_hat, xpx_Omegainv, IW_param] = posterior_niw(Y, p, Psi, b, Omega);

% Draw
normal_draws = mvnrnd(zeros(n,1), inv(IW_param), T-p+d);
Sigma_draw = inv(normal_draws'*normal_draws);
beta_draw = mvnrnd(B_hat(:), kron(Sigma_draw, inv(xpx_Omegainv)));
B_draw = reshape(beta_draw, [], n);

end