function [x,error_me_sgd_noise] = acc_me_sgd(s_k,f,gradf,gradf_i,x0,iter,s,c,d,x_star)

x=x0;
v=x0;
n = length(x);
t_k_sum=0;
L = 1/(c^2);
for k=1:iter
    %error_me(k) = norm(gradf(x(:,k)))^2;
    %error_me1(mont,k) = min(error_me(mont,(1:k)));
    error_me_sgd_noise(1,k) = f(x)-f(x_star);
        t_k_sum = t_k_sum+s_k(k);
        noise1 = randn(n,1);
        sigma=0;
        gard_c = gradf(x);


        for j = 1:d
            sigma = sigma +(gard_c-gradf_i(x,j))* (gard_c-gradf_i(x,j))';
        end
        sigma = sigma./d;
        x = (1+2*s_k(k)/t_k_sum)^(-1)*(x + 2*s_k(k)/t_k_sum *(v))-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)/sqrt(L)*(gradf(x)+sigma^(1/2)*noise1);
        


        sigma=0;
        gard_c = gradf(x);
        for j = 1:d
            sigma = sigma +(gard_c-gradf_i(x,j))* (gard_c-gradf_i(x,j))';
        end
        sigma = sigma./d;
        noise1 = randn(n,1);
        v = v - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*s_k(k))*(gradf(x)+sigma^(1/2)*noise1);

    
end


end