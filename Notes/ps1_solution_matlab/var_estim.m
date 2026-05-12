function [betahat, Sigmahat, res, X] = var_estim(Y, p)

    % VAR(p) least-squares estimator

    T = size(Y,1)-p;
    Y_lag = lagmatrix(Y,1:p);
    X = [ones(T,1) Y_lag(p+1:end,:)];
    
    betahat = (X\Y(p+1:end,:))';
    
    res = Y(p+1:end,:) - X*betahat';
    Sigmahat = (res'*res)/T; 

end