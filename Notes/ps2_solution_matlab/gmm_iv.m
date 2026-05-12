function [betahat, betahat_var, Omegahat, J_stat] = gmm_iv(Y, X, Z, W, hac_fct, eff)

    % Linear GMM IV estimator w. HAC variance
    % Also computes over-ID statistic given efficient weight matrix

    % Point estimator
    S_ZX = Z'*X;
    S_ZY = Z'*Y;
    denom = S_ZX'*W*S_ZX;
    betahat = denom\(S_ZX'*W*S_ZY); % GMM estimator
    
    res = Y-X*betahat; % Residuals
    
    % HAC sandwich matrix
    Omegahat = hac_fct(Z.*res);
    
    T = length(Y);
    
    if eff % If efficient weight matrix...
        
        betahat_var = T*inv(S_ZX'*(Omegahat\S_ZX)); % Efficient variance
        S_Zres = Z'*res;
        J_stat = (S_Zres'*(Omegahat\S_Zres))/T; % J test statistic
        
    else % If not efficient weight matrix...
        
        betahat_var = T*(denom\((S_ZX'*W*Omegahat*W*S_ZX)/denom)); % Variance
        J_stat = []; % Can't compute J test statistic w/o efficient weight matrix
        
    end
    

end