clc
clear


d=10;
n=5;
alpha=3/4;
Alpha=3/4;
m=0;
nepochs = 1;
iter = 1e5;
mont_iter = 1;
%% Vector Norm Squared function
A = randn(d,n);
%A = A' * A;
% [U , Eig , V] = svd(A);
% A = A - U(:,n)*Eig(n,n)*V(n,:);
x_star=randn(n,1)+2;
b = A*x_star;
f = @(x) 1/(d*2)*norm(A*x-b)^2;
gradf = @(x) 1/d*A'*(A*x-b); 
gradf_i = @(x,i) 1/d*A(i,:)'*(A(i,:)*x-b(i)); 
%% Stochastic version of HR Nag
% First I find the optimal values
L = max(eig(A' * A));
c=1/sqrt(L);
s=1/L;
f_star = 0;

%% Main implementation
s_k =@(k) c/(k^alpha);
s_k1 =@(k) c/2 +c/(2*k^Alpha);
%t_k =@(p) c*p^(1-alpha)/(1-alpha);

for mont=1:mont_iter
%% Gaussian Noise

 x0=randn(n,1);
 sigma=1;
 [x_me_gauss(:,:,mont),error_me_Gaus_noise(mont,:),upperbound,upperbound_laborde] = acc_gaussian(s_k,f,gradf,x0,iter*nepochs,s,c,sigma,x_star);

%% NNAG + SGD Noise
%[~,error_me_sgd_noise(mont,:)] = acc_me_sgd(s_k,f,gradf,gradf_i,x0,iter,s,c,d,x_star);


%% NNAG + SVRG Noise
m=10;
%[~,error_me_svrg_noise(mont,:)] = acc_me_svrg(s_k,f,gradf,gradf_i,x0,iter,s,c,d,m,x_star);

%% Laborde + SGD Noise
 %[~,error_laborde1(mont,:)] = acc_laborde_sgd_noise(s_k,f,gradf,gradf_i,x0,iter,s,c,d,m,x_star);
%% SGD
[x_sgd,error_sgd(mont,:),std_rate] = sgd(s_k,f,gradf,gradf_i,x0,iter*nepochs,s,c,d,m,x_star);
%% Laborde + Gaussian Noise
%% SGD Version of my method
% [~, error_me_sgd(mont,:)] = my_update_with_sgd_grad(s_k,f,gradf,gradf_i,x0,iter,s,c,d,m,x_star);
end

%% GD
%  x1=x(:,1) ;
%  v1=x1;
% for i=1:iter
% error_GD(i) = f(x1)-f_star;
% x1 = x1 - s*gradf(x1);
% 
% end

loglog(1/mont_iter*sum(error_me_Gaus_noise,1))
hold on
%loglog(1/mont_iter*sum(error_me_svrg_noise,1))
%loglog(1/mont_iter*sum(error_me_sgd_noise,1))
%loglog(1/mont_iter*sum(error_laborde1,1))
loglog(1/mont_iter*sum(error_sgd,1))
 loglog(1/mont_iter*sum(std_rate,1))
%loglog(1/mont_iter*sum(error_me_sgd,1))
%loglog(error_GD)
%sum_bound_me = 1/mont_iter*sum(bound_me,1);
%loglog(sum_bound_me)
%sum_bound_laborde = 1/mont_iter*sum(bound_laborde,1);
%loglog(sum_bound_laborde)
%plot(sum_bound_laborde)