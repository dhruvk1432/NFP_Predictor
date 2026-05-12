clear;

% ***************************
% ECON 31730 Problem Set 2
% Solution to Q5
% Mikkel Plagborg-Moller
% 2026-04-30
% ***************************


%% Settings

% AR settings
p_max = 50;
omega_vals_ar = linspace(0,pi,100);

% Kernel settings
kernel = @(x) max(1-x.^2,0);
bandwidth = 10;

% Plot settings
signif_level = 0.05;
set_ylim = [-8 4];
the_title = {'Not seasonally adjusted', 'Seasonally adjusted'};


%% Load data

dat = readtable('monthly.csv');
series = {'UNRATENSA', 'UNRATE'};
Y = dat{:,series};
n_Y = size(Y,2);


%% AR spectrum

p_hats = zeros(1,n_Y);
spec_dens_ar = zeros(length(omega_vals_ar),n_Y);
spec_dens_se_ar = zeros(length(omega_vals_ar),n_Y);

for j=1:n_Y
    
    disp('Series');
    disp(series{j});
    
    % Select lag length
    the_bics = ar_ic(Y(:,j),p_max);
    [~,the_min_ind] = min(the_bics);
    the_p_hat = the_min_ind;
    disp('hat{p}');
    disp(the_p_hat);
    
    % Estimate AR(p)
    [the_beta_hat, the_sigma2_hat, the_beta_hat_var, the_sigma2_hat_var] = ar_estim(Y(:,j),the_p_hat);
    
    % Spectral density and standard errors
    [the_spec_dens, the_spec_dens_se] = ar_spec(the_beta_hat(2:end), the_sigma2_hat, omega_vals_ar, the_beta_hat_var(2:end,2:end), the_sigma2_hat_var);
    
    % Store results
    p_hats(j) = the_p_hat;
    spec_dens_ar(:,j) = the_spec_dens';
    spec_dens_se_ar(:,j) = the_spec_dens_se';
    
end

% Plot
cv = norminv(1-signif_level/2);
figure('Unit', 'normalize', 'Position', [0.1 0.2 0.8 0.6]);
for j=1:n_Y
    subplot(1,n_Y,j);
    plot(omega_vals_ar/(2*pi),spec_dens_ar(:,j),'-k',...
         omega_vals_ar/(2*pi),spec_dens_ar(:,j)-cv*spec_dens_se_ar(:,j),'-b',...
         omega_vals_ar/(2*pi),spec_dens_ar(:,j)+cv*spec_dens_se_ar(:,j),'-b');
    ylim(set_ylim);
    xlabel('frequency/(2*pi)');
    ylabel('log spectrum');
    set(gca,'FontSize',12);
    title(the_title{j},'FontSize',14);
end


%% Kernel estimation

% Smooth
omega_vals_kernel = cell(1,n_Y);
log_spec_dens_kernel = cell(1,n_Y);
log_spec_dens_se_kernel = cell(1,n_Y);

for j=1:n_Y
    
    [the_omega_vals, the_log_spec_dens, the_log_spec_dens_se] = kernel_spec(Y(:,j), kernel, bandwidth);
    omega_vals_kernel{j} = the_omega_vals;
    log_spec_dens_kernel{j} = the_log_spec_dens;
    log_spec_dens_se_kernel{j} = the_log_spec_dens_se;
    
end

% Plot
figure('Unit', 'normalize', 'Position', [0.1 0.2 0.8 0.6]);
for j=1:n_Y
    subplot(1,n_Y,j);
    plot(omega_vals_kernel{j}/(2*pi),log_spec_dens_kernel{j},'-k',...
         omega_vals_kernel{j}/(2*pi),log_spec_dens_kernel{j}-cv*log_spec_dens_se_kernel{j},'-b',...
         omega_vals_kernel{j}/(2*pi),log_spec_dens_kernel{j}+cv*log_spec_dens_se_kernel{j},'-b');
    ylim(set_ylim);
    xlabel('frequency/(2*pi)');
    ylabel('log spectrum');
    set(gca,'FontSize',12);
    title(the_title{j},'FontSize',14);
end

