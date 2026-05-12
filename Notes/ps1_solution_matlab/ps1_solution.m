clear;

% ***************************
% ECON 31730 Problem Set 1
% Solutions to Q3+4
% Mikkel Plagborg-Moller
% 2026-04-14
% ***************************


%% Load data

dat_monthly = table2timetable(readtable('monthly.csv'));
dat_quarterly = table2timetable(readtable('quarterly.csv'));

dat_quarterly.rgdp_growth = [NaN; 4*log(dat_quarterly{2:end,'GDPC1'}./dat_quarterly{1:end-1,'GDPC1'})]; % RGDP log growth, annualized
dat_quarterly.pgdp_growth = [NaN; 4*log(dat_quarterly{2:end,'GDPDEF'}./dat_quarterly{1:end-1,'GDPDEF'})]; % PGDP log growth, annualized

dat = innerjoin(retime(dat_monthly,'quarterly','mean'), dat_quarterly);

rgdp_growth = dat.rgdp_growth;
pgdp_growth = dat.pgdp_growth;
fedfunds_real = 0.01*dat{:,'FEDFUNDS'}-pgdp_growth; % Fed Funds Rate minus inflation

% Keep relevant sample
time = datetime(dat.Properties.RowTimes);
sample = (time >= datetime('1954-07-01')) & (time <= datetime('2019-12-31'));
time = time(sample);
rgdp_growth = rgdp_growth(sample);
pgdp_growth = pgdp_growth(sample);
fedfunds_real = fedfunds_real(sample);

Y = [rgdp_growth pgdp_growth fedfunds_real]; % VAR variables;


%% Question 3

ps = 0:20; % Lag orders to consider
[bic, aic] = var_ic(Y, ps);
[~,min_ind_bic] = min(bic);
[~,min_ind_aic] = min(aic);

% Display results
disp('*** Q3');
disp('Optimal p [BIC AIC]:');
disp(ps([min_ind_bic, min_ind_aic]));


%% Question 4(i)

% Define prior hyperparameters
n = size(Y,2); % VAR dimension
d = n+2;
Psi = (d-n-1)*0.02^2*eye(n);
cov_scale = 0.2^2/(0.02^2/(d-n-1)); % Used in var-cov specification
Omega = @(p) diag([10^2, cov_scale*repmat((1:p).^(-2),1,n)]); % Returns prior hyperparameter Omega, given p

% Compute log marginal likelihood for each p
disp('*** Q4');
disp('Log marginal likelihood values');
lmls = zeros(size(ps));
for j=1:length(ps)
    
    % Log marginal likelihood (enforce same estimation sample for different p)
    lmls(j) = logmarglik_niw(Y(max(ps)-ps(j)+1:end,:), ps(j), ...
                             Psi, d, zeros(n*(n*ps(j)+1),1), Omega(ps(j)));
    
    % Display results
    fprintf('%2d%12.2f\n', ps(j), lmls(j));
    
end

% Optimal p according to marginal likelihood
[~,max_ind] = max(lmls);
p_maxml = ps(max_ind);

% Display results
disp('Optimal p')
disp(p_maxml);

% Plot figure
figure;
plot(ps, lmls, '-o');
title('log marginal likelihood');
xlabel('p');


%% Question 4(ii)

% Produce 10,000 predictive draws of y_{T+2}
y_pred_draws = predictive_draws_niw(Y, p_maxml, Psi, d, zeros(n*(n*p_maxml+1),1), Omega(p_maxml), 2, 1e4);

% Kernel density estimate of posterior predictive density of y_{1,T+2}
[f,xi] = ksdensity(y_pred_draws(:,1));

% Plot figure
figure;
plot(xi, f, '-');
title('Predictive density of y_{1,T+2}');
