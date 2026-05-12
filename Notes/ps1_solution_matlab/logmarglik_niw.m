function lml = logmarglik_niw(Y, p, Psi, d, b, Omega)

% Computes log marginal likelihood for VAR(p) model
% given data Y and normal-inverse-Wishart prior

% See notation and formulas in Giannone, Lenza & Primiceri (REStat 2015)
% and their online appendix


% Dimensions
[T,n] = size(Y);

% Formula components
[~, xpx_Omegainv, IW_param] = posterior_niw(Y, p, Psi, b, Omega);

% Log marginal likelihood
lml =   -0.5*n*(T-p)*log(pi) ...
        + logMvGamma(0.5*(T-p+d),n) ...
        - logMvGamma(0.5*d,n) ...
        - 0.5*n*log(det(Omega)) ...
        + 0.5*d*log(det(Psi)) ...
        - 0.5*n*log(det(xpx_Omegainv)) ...
        - 0.5*(T-p+d)*log(det(IW_param));

end