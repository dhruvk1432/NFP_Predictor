clear;

% ***************************
% ECON 31730 Problem Set 2
% Solution to Q6
% Mikkel Plagborg-Moller
% 2026-04-30
% ***************************


%% Settings

Z_lags = 3; % Number of IV lags of inflation and unemployment rate


%% Load data

% Load from data files
dat_monthly = table2timetable(readtable('monthly.csv'));
dat_quarterly = table2timetable(readtable('quarterly.csv'));
dat = innerjoin(retime(dat_monthly,'quarterly','mean'), dat_quarterly);

% Read data series
price = dat.GDPDEF;
unrate = dat.UNRATE;
infl = [NaN; 100*log(price(2:end)./price(1:end-1))];

% Enforce common sample for data series
data_matrix = [infl unrate];
sample = find(all(~isnan(data_matrix),2),1):find(all(~isnan(data_matrix),2),1,'last');
data_matrix_sample = data_matrix(sample,:);

% Response variable: inflation
Y = data_matrix_sample(Z_lags+1:end-1,1);

% Regressors: constant, future inflation, lagged inflation, unemployment rate
T = length(Y);
X = [ones(T,1) data_matrix_sample(Z_lags+2:end,1) data_matrix_sample(Z_lags:end-2,1) data_matrix_sample(Z_lags+1:end-1,2)];

% Instruments: constant, lags of inflation and unemployment rate
data_matrix_lag = lagmatrix(data_matrix_sample,1:Z_lags);
Z = [ones(T,1) data_matrix_lag(Z_lags+1:end-1,:)];

% Dimensions
k = size(X,2);
r = size(Z,2);


%% GMM estimation

% 1st step
[betahat_1st, betahat_var_1st, Omegahat_1st] = gmm_iv(Y, X, Z, inv(Z'*Z), @newey_west, false);

% 2nd (efficient) step
[betahat_eff, betahat_var_eff, ~, J_stat] = gmm_iv(Y, X, Z, inv(Omegahat_1st), @newey_west, true);

% J test p-value
J_pval = 1-chi2cdf(J_stat, r-k);


%% Report results

disp('1st step: point estimates [const pi(t+1) pi(t-1) x(t)]');
disp(betahat_1st');

disp('1st step: standard errors');
disp(sqrt(diag(betahat_var_1st)'));

disp('2nd step: point estimates');
disp(betahat_eff');

disp('2nd step: standard errors');
disp(sqrt(diag(betahat_var_eff)'));

disp('J test: statistic');
disp(J_stat);

disp('J test: p-value');
disp(J_pval);

