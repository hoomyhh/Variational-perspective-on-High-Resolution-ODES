function [x,error_me_svrg_noise] = acc_me_svrg(s_k,f,gradf,gradf_i,x0,iter,s,c,d,m,x_star)

x = x0;
x_hat = x;
v = x;
L = 1/(c^2);
gard_c = gradf(x);
t_k_sum = 0;
n = length(x);
for k=1:iter
    if rem(k,m)==0
       x_hat = x;
    end
    %error_me(mont,k) = norm(gradf(x(:,k)))^2;
    %error_me1(mont,k) = min(error_me(mont,(1:k)));
    error_me_svrg_noise(1,k) = norm(x-x_star)^2;
        t_k_sum = t_k_sum+s_k(k);
        noise1 = randn(n,1);

        sigma=0;
        for j = 1:d 
            gard_c_i = gradf_i(x,j);
            sigma = sigma +(gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x))* (gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x))';
        end
        sigma = sigma./d;
        x = (1+2*s_k(k)/t_k_sum)^(-1)*(x + 2*s_k(k)/t_k_sum *(v))-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)/sqrt(L)*(gradf(x)+sigma^(1/2)*noise1);
        


        sigma=0;
        for j = 1:d 
            gard_c_i = gradf_i(x,j);
            sigma = sigma +(gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x))* (gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x))';
        end
        sigma = sigma./d;
        noise1 = randn(n,1);
        v = v - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*s_k(k))*(gradf(x)+sigma^(1/2)*noise1);

    norm_sigma(k)=norm(sigma,'fro');
end

end