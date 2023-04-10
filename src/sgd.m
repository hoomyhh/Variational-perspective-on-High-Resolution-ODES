function [x1_save,error_sgd,std_rate] = sgd(s_k,f,gradf,gradf_i,x0,iter,s,c,d,m,x_star)
L = 1/(c^2);
x1=x0 ;
v1=x1;
eta = 0.2;
beta = 1/L;
for i=1:iter
    x1_save(:,i) = x1;
    j=randi(d);    
    %x1 = x1 - beta*gradf_i(x1,j)-eta*(v1-x1);
    x1 = x1 - beta*gradf_i(x1,j);
    v1=x1_save;
    error_sgd(1,i)=f(x1)-f(x_star);
    std_rate(1,i) = f(x0)-f(x_star)/sqrt(i);
end
end