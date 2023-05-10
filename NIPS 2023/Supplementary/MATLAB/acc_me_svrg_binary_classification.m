function [x1_save,error_srvg] = acc_me_svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star,f,d)
L = 1/(c^2);
x1=x0 ;
v1=x0;
c = 1/(sqrt(L));
alpha = 3/4;
s_k =@(k) c/(k^alpha);
a= L/10;
[n,~]=size(X);
for k=1:nepochs
    xhat = x1;
    ghat = zeros(d,1);
    for p =1:n
        xi = X(p,:);
        yi = Y(p);
        ghat = ghat + gradf_i(xhat,xi,yi);
    end
    ghat = ghat/n;
     
t_k_sum=0; 
for i=1:iter
    error_srvg(1,(k-1)*iter+i)=f(x1,X,Y)-f(x_star,X,Y);
    t_k_sum = t_k_sum+s_k(i);

    x1_save(:,(k-1)*iter+i) = x1;
    j=randi(d); 
    xi = X(j,:);
    yi = Y(j);
    x1 = (1+2*s_k(i)/t_k_sum)^(-1)*(x1 + 2*s_k(i)/t_k_sum *v1)-(1+2*s_k(i)/t_k_sum)^(-1)*s_k(i)*a/sqrt(L)*(gradf_i(x1,xi,yi)-gradf_i(xhat,xi,yi)+ghat);
    v1 = v1 - 1/2*(t_k_sum*s_k(i)+2*s_k(i)*a/sqrt(L))*(gradf_i(x1,xi,yi)-gradf_i(xhat,xi,yi)+ghat);

end
end
end