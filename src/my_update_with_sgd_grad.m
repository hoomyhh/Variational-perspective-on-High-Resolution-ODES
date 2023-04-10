function [x, error_me_sgd] = my_update_with_sgd_grad(s_k,f,gradf,gradf_i,x0,iter,s,c,d,m,x_star)

t_k_sum = 0;

x=x0;
v=x;
L = 1/(c^2);
for k=1:iter
    %error_me(mont,k) = norm(gradf(x))^2;
    %error_me1(mont,k) = min(error_me(mont,(1:k)));
    error_me_sgd(1,k) = norm(x-x_star)^2;
        t_k_sum = t_k_sum+s_k(k);
        j=randi(d);
        
        x = (1+2*s_k(k)/t_k_sum)^(-1)*(x + 2*s_k(k)/t_k_sum *(v))-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)/sqrt(L)*(gradf_i(x,j));
       
        v = v - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*s_k(k))*(gradf_i(x,j));

    
end
end