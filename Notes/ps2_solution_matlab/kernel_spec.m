function [omega_vals, log_spec_dens, log_spec_dens_se] = kernel_spec(Y, kernel, bandwidth)

    % Kernel estimation of log spectral density

    % Compute periodogram ordinates
    T = length(Y);
    periodogram = (abs(fft(Y)).^2/T);
    periodogram(1) = periodogram(2); % Handle frequency zero
    
    % Stack periodogram so that we can apply smoothing algorithm
    T_mid = (T-mod(T,2))/2; % Mid point of sample (depends on whether T even/odd)
    periodogram_stack = [periodogram(end-T_mid+1:end); periodogram];
    
    % Smooth the stacked periodogram
    diff_ind = abs((1:T+T_mid)'-(1:T+T_mid));
    weights = kernel(diff_ind/bandwidth).*(diff_ind <= T_mid); % Weights
    weights_norm = bsxfun(@rdivide, weights, sum(weights,2)); % Normalize weights to sum to 1
    smooth_stack = weights_norm*periodogram_stack/(2*pi); % Apply smoother
    
    % Outputs
    omega_vals = 2*pi*((0:T_mid)/T)'; % Fourier frequencies
    log_spec_dens = log(smooth_stack(T_mid+1:2*T_mid+1)); % Log spectral density
    log_spec_dens_se = sqrt(sum(weights_norm(T_mid+1:2*T_mid+1,:).^2,2)); % S.e. of log spectral density
    

end