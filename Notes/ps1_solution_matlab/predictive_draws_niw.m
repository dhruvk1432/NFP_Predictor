function y_pred_draws = predictive_draws_niw(Y, p, Psi, d, b, Omega, horizon, numdraws)

% Generates "numdraws" posterior predictive draws of Y_{T+horizon} for
% VAR(p) model given data Y and normal-inverse-Wishart prior

% See notation and formulas in Giannone, Lenza & Primiceri (REStat 2015)
% and their online appendix


% Dimensions
[T,n] = size(Y);

y_pred_draws = zeros(numdraws,n);

for j=1:numdraws
   
    % Posterior parameter draw
    [the_B_draw, the_Sigma_draw] = posterior_draw_niw(Y, p, Psi, d, b, Omega);
    the_C_draw = the_B_draw(1,:)'; % Intercept
    the_Blag_draw = reshape(the_B_draw(2:end,:)',n,n,p); % Lag coefficients
    
    % Simulate future values using VAR equation
    the_Y_pad = [Y; zeros(horizon,n)];
    for m=1:horizon
        the_Y_new = mvnrnd(the_C_draw, the_Sigma_draw); % Constant plus innovation
        for l=1:p
            the_Y_new = the_Y_new + the_Y_pad(T+m-l,:)*the_Blag_draw(:,:,l)'; % Add lags
        end
        the_Y_pad(T+m,:) = the_Y_new; % Record simulated new observation
    end
    
    y_pred_draws(j,:) = the_Y_pad(end,:); % Store last simulated observation
    
end

end