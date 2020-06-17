function [mu,sigma] = logtrans(mu_norm,sig_norm)

    mu = log((mu_norm^2)/sqrt(sig_norm^2+mu_norm^2));
    sigma = sqrt(log(sig_norm^2/(mu_norm^2)+1));


end