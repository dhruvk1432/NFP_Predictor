function [beta_hat, sigma2_hat, beta_hat_var, sigma2_hat_var] = ar_estim(Y,p)
    
    % Least-squares AR(p) estimation

    T = length(Y);
    Y_lag = lagmatrix(Y,1:p);
    X = [ones(T-p,1) Y_lag(p+1:end,:)];
    beta_hat = X\Y(p+1:end);
    res = Y(p+1:end)-X*beta_hat;
    sigma2_hat = var(res,1);
    
    % Asymptotic variances
    
    if nargout>2
        
        beta_hat_var = sigma2_hat*inv(X'*X);
        sigma2_hat_var = var(res.^2)/(T-p);
        
    end

end