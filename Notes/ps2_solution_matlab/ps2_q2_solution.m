clear;

% ***********************************************
% ECON 31730 Problem Set 2
% Solution to Q2
% Mikkel Plagborg-Moller
% 2026-04-30
% ***********************************************


%% Settings

beta_true = [0.6 0.2];  % True AR(p) coefs
sigma2_true = 1;        % True innovation variance
T = 200;                % Sample size
T_burnin = 200;         % Burn-in sample size for simulating AR data

p_max = 3;              % Maximum number of lags for IC (minimum: 1)
numrep = 1e4;           % Number of simulation replications
rng(20170917);          % Seed RNG


%% Compute roots of AR polynomial

disp('Roots of AR polynomial');
disp(roots([-beta_true(end:-1:1) 1]));


%% Run simulation

rho1_hats = zeros(numrep,2); % 1st column: BIC, 2nd column: AIC
p_hats = zeros(numrep,2); 

for n=1:numrep
    
    % Simulate data and throw away burn-in sample
    the_Z = sqrt(sigma2_true)*randn(T+T_burnin,1);
    the_Y = filter(1, [1 -beta_true], the_Z);
    the_Y = the_Y(end-T+1:end);
    
    % Compute information criteria
    [the_bics,the_aics] = ar_ic(the_Y,p_max);
    
    % hat{p}
    [~,the_p_hat_bic] = min(the_bics);              
    [~,the_p_hat_aic] = min(the_aics);
    
    % AR estimators, given hat{p}
    the_beta_hat_bic = ar_estim(the_Y, the_p_hat_bic);
    the_beta_hat_aic = ar_estim(the_Y, the_p_hat_aic);
    
    % Store results
    p_hats(n,:) = [the_p_hat_bic the_p_hat_aic];
    rho1_hats(n,:) = [the_beta_hat_bic(2) the_beta_hat_aic(2)]; % AR coef on 1st lag
    
end


%% Plot sampling density

% Kernel density estimates
[f_bic,xi_bic] = ksdensity(rho1_hats(:,1));
[f_aic,xi_aic] = ksdensity(rho1_hats(:,2));

% Sampling distribution of hat{rho}_1
figure('Unit', 'normalize', 'Position', [0.2 0.2 0.6 0.6]);
plot(xi_bic, f_bic, '-k', xi_aic, f_aic, '--r', 'LineWidth', 2);
the_ylim = ylim;
line(beta_true(1)*ones(1,2), the_ylim, 'Color', 'k', 'LineStyle', '--');
ylim(the_ylim);
set(gca, 'FontSize', 14);
legend({'BIC', 'AIC'}, 'FontSize', 14);

% Sampling distribution of hat{p}
figure('Unit', 'normalize', 'Position', [0.2 0.2 0.6 0.6]);

subplot(1,2,1);
histogram(p_hats(:,1), 'Normalization', 'probability');
set(gca, 'FontSize', 14);
set(gca, 'XTick', 1:p_max);
ylim([0 1]);
title('BIC', 'FontSize', 14);

subplot(1,2,2);
histogram(p_hats(:,2), 'Normalization', 'probability');
set(gca, 'FontSize', 14);
set(gca, 'XTick', 1:p_max);
ylim([0 1]);
title('AIC', 'FontSize', 14);


