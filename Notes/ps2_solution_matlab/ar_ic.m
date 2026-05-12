function [bics, aics] = ar_ic(Y,p_max)

    % Information criteria (BIC+AIC) for AR(p)

    % Residual variance for each p
    sigma2_hats = zeros(p_max,1);
    for p=1:p_max
        [~,the_sigma2_hat] = ar_estim(Y(p_max-p+1:end),p);
        sigma2_hats(p) = the_sigma2_hat;
    end
    
    T = length(Y);
    bics = log(sigma2_hats) + log(T)/T*(1:p_max)'; % BIC
    aics = log(sigma2_hats) + 2/T*(1:p_max)'; % AIC

end