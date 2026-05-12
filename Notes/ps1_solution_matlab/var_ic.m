function [bic, aic] = var_ic(Y, ps)

    % Compute BIC and AIC for VAR(p) model

    [T,n] = size(Y);
    num_p = length(ps);
    p_max = max(ps);
    Teff = T-p_max; % Effective sample size

    bic = nan(size(ps));
    aic = nan(size(ps));

    for i_p = 1:num_p % For each lag length p...

        p = ps(i_p);
        [~,the_Sigmahat] = var_estim(Y(p_max-p+1:end,:), p); % Estimate VAR(p) (enforce same estimation sample for different p)
        the_logdet = log(det(the_Sigmahat)); % log(det(Sigma))

        bic(p+1) = the_logdet + n^2*p*log(Teff)/Teff; % BIC
        aic(p+1) = the_logdet + n^2*p*2/Teff; % AIC

    end

end