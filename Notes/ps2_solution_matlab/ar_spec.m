function [log_spec_dens, log_spec_dens_se] = ar_spec(coefs, sigma2, omega_vals, varargin)

    % Compute AR(p) log spectrum

    p = length(coefs);
    spec_dens_denom = abs(1-coefs'*exp(-1j*(1:p)'*omega_vals)).^2;
    log_spec_dens = log(sigma2/(2*pi)) - log(spec_dens_denom);
    
    if nargout>1
        
        % Compute delta method variance of AR(p) log spectrum
        
        log_spec_dens_se = zeros(size(log_spec_dens));
        
        coefs_var = varargin{1}; % Var-cov matrix of AR coefficients
        sigma2_var = varargin{2}; % Variance of sigma^2
        
        for w=1:length(omega_vals) % At each frequency...
            
            % Compute derivative of denominator wrt. AR coefficients
            the_deriv_coefs = -2*cos(omega_vals(w)*(1:p)) + 2*coefs'*cos(omega_vals(w)*((1:p)'-(1:p)));
            
            % Variance using delta method
            the_var =   sigma2^(-2)*sigma2_var ...
                        + spec_dens_denom(w)^(-2)*the_deriv_coefs*coefs_var*the_deriv_coefs';
                    
            % Standard errors
            log_spec_dens_se(w) = sqrt(the_var);
            
        end
        
    end

end