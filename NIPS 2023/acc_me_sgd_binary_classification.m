function [x1_save,error_sgd,std_rate] = acc_me_sgd_binary_classification(X,Y,s_k,f,gradf_i,x0,iter,c,n,x_star,L)
x1=x0 ;
v1=x1;
a = L;
t_k_sum=0; 
for i=1:iter
    error_sgd(1,i)=f(x1,X,Y)-f(x_star,X,Y);
    t_k_sum = t_k_sum+s_k(i);        
    x_save(:,i)=x1;
    
    x1_save(:,i) = x1;
    j=randi(n);   
    xi = X(j,:);
    yi = Y(j);

    x1 = (1+2*s_k(i)/t_k_sum)^(-1)*(x1 + 2*s_k(i)/t_k_sum *v1)-(1+2*s_k(i)/t_k_sum)^(-1)*s_k(i)*a/sqrt(L)*(gradf_i(x1,xi,yi));
    v1 = v1 - 1/2*(t_k_sum*s_k(i)+2*s_k(i)*a/sqrt(L))*(gradf_i(x1,xi,yi));
    std_rate(1,i) = (f(x0,X,Y)-f(x_star,X,Y))/sqrt(i);
end
end