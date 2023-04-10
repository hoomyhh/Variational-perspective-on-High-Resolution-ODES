clc
clear

rng(20)
d=10;
n=1000;
alpha=3/4;
Alpha=2/3;
m=0;
nepochs = 100;
iter = 1e3;
mont_iter = 100;
%% Vector Norm Squared function
x_star = randn(d,1);
X = randn(n,d)+1;

Y = (1./(1+exp(-X*x_star))>= rand(n,1))*2-1;
f = @(x,X,Y) 1/n*sum(log(1+exp(-(Y.*X)*x)));
gradf_i = @(x,xi,yi) -xi'*yi*exp(-yi*xi*x)/(1+exp(-yi*xi*x));

%% Stochastic version of HR Nag
% First I find the optimal values
x0 = randn(d,1);
L = max(Y.^2.*diag(X*X'));
c=1/sqrt(L);
c1 = 1/L;
%s=1/L;
%[x_opt,error_srvg] = svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star);


%% Main implementation
s_k =@(k) c/(k^alpha);
s_k1 =@(k) c1/(k^Alpha);
%t_k =@(p) c*p^(1-alpha)/(1-alpha);

for mont=1:mont_iter
%% Gaussian Noise
 x0=randn(d,1);
 sigma=1;
[x_me(:,:,mont),error_me_Gaus(mont,:),upperbound,upperbound2,upperbound_laborde] = acc_gaussian_binary_class(s_k,nepochs,x0,iter*nepochs,c,sigma,x_star,X,Y,f);
%% SGD Noise
% t_k_sum = 0;
% 
% x(:,1)=x(:,1);
% v(:,1)=x(:,1);
% 
% for k=1:iter
%     error_me(mont,k) = norm(gradf(x(:,k)))^2;
%     %error_me1(mont,k) = min(error_me(mont,(1:k)));
%     error_me_sgd_noise(mont,k) = f(x(:,k))-f_star;
%         t_k_sum = t_k_sum+s_k(k);
%         noise1 = randn(n,1);
%         sigma=0;
%         gard_c = gradf(x(:,k));
%         for j = 1:d
%             %Z = zeros(n,1);
%             %Z(j) = gard_c(j);
%             sigma = sigma +(gard_c-gradf_i(x(:,k),j))* (gard_c-gradf_i(x(:,k),j))';
%         end
%         sigma = sigma./d;
%         
%         x(:,k+1) = (1+2*s_k(k)/t_k_sum)^(-1)*(x(:,k) + 2*s_k(k)/t_k_sum *(v(:,k)))-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)/sqrt(L)*(gradf(x(:,k))+sigma^(1/2)*noise1);
%         
%         sigma=0;
%         gard_c = gradf(x(:,k+1));
%         for j = 1:d
% %             Z = zeros(n,1);
% %             Z(j) = gard_c(j);
%             sigma = sigma +(gard_c-gradf_i(x(:,k+1),j))* (gard_c-gradf_i(x(:,k+1),j))';
%         end
%         sigma = sigma./d;
%         noise1 = randn(n,1);
%         v(:,k+1) = v(:,k) - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*s_k(k))*(gradf(x(:,k+1))+sigma^(1/2)*noise1);
% 
%     
% end
%% SVRG Noise
% t_k_sum = 0;
% m=10;
% x(:,1)=x(:,1);
% x_hat = x(:,1);
% v(:,1)=x(:,1);
% gard_c = gradf(x(:,1));
% for k=1:iter
%     if rem(k,m)==0
%        x_hat = x(:,k);
%     end
%    
%     error_me(mont,k) = norm(gradf(x(:,k)))^2;
%     %error_me1(mont,k) = min(error_me(mont,(1:k)));
%     error_me_svrg_noise(mont,k) = f(x(:,k))-f_star;
%         t_k_sum = t_k_sum+s_k(k);
%         noise1 = randn(n,1);
%         sigma=0;
%         
%         for j = 1:d 
%             gard_c_i = gradf_i(x(:,k),j);
%             %Z = zeros(n,1);
%             %Z(j) = gard_c(j);
%             sigma = sigma +(gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x(:,k)))* (gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x(:,k)))';
%         end
%         sigma = sigma./d;
%         
%         x(:,k+1) = (1+2*s_k(k)/t_k_sum)^(-1)*(x(:,k) + 2*s_k(k)/t_k_sum *(v(:,k)))-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)/sqrt(L)*(gradf(x(:,k))+sigma^(1/2)*noise1);
%         
%         sigma=0;
%         for j = 1:d 
%             gard_c_i = gradf_i(x(:,k+1),j);
%             %Z = zeros(n,1);
%             %Z(j) = gard_c(j);
%             sigma = sigma +(gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x(:,k+1)))* (gard_c_i-gradf_i(x_hat,j)+gradf(x_hat)-gradf(x(:,k+1)))';
%         end
%         sigma = sigma./d;
%         noise1 = randn(n,1);
%         v(:,k+1) = v(:,k) - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*s_k(k))*(gradf(x(:,k+1))+sigma^(1/2)*noise1);
% 
%     norm_sigma(k)=norm(sigma,'fro');
% end
%% Laborde
[x_laborde(:,:,mont),error_laborde1(mont,:)] = acc_gaussian_laborde(f,x0,X,Y,c,s_k,iter*nepochs,x_star);
%% SGD
 [x_sgd_noise(:,:,mont),x_sgd(:,:,mont),error_sgd(mont,:),std_rate,error_sgd2(mont,:)] = sgd_binary_classification(X,Y,s_k1,gradf_i,x0,iter*nepochs,c,x_star,f);

%% SGD Version of my method
% t_k_sum = 0;
% 
% x(:,1)=x(:,1);
% v(:,1)=x(:,1);
% 
% for k=1:iter
%     error_me(mont,k) = norm(gradf(x(:,k)))^2;
%     %error_me1(mont,k) = min(error_me(mont,(1:k)));
%     error_me_sgd(mont,k) = f(x(:,k))-f_star;
%         t_k_sum = t_k_sum+s_k(k);
%         j=randi(d);
%         
%         x(:,k+1) = (1+2*s_k(k)/t_k_sum)^(-1)*(x(:,k) + 2*s_k(k)/t_k_sum *(v(:,k)))-(1+2*s_k(k)/t_k_sum)^(-1)*s_k(k)/sqrt(L)*(gradf_i(x(:,k),j));
%        
%         v(:,k+1) = v(:,k) - 1/2*(t_k_sum*s_k(k)+2*s_k(k)*s_k(k))*(gradf_i(x(:,k+1),j));
% 
%     
% end
%% SVRG
[x_svrg(:,:,mont),error_svrg(mont,:)] = svrg_binary_classification(X,Y,gradf_i,x0,iter,c,nepochs,x_star,f);

mont
end

%% GD
 x1=x0 ;
 v1=x1;
for i=1:iter*nepochs
error_GD(i) = f(x1,X,Y)-f(x_star,X,Y);
x1 = x1 - 1/L*sum(-1/n.*(Y.*X)'.*(exp(-Y.*(X*x1)))'./(1+exp(-Y.*(X*x1)))',2);
end
 
    loglog(1/mont_iter*sum(error_me_Gaus,1))
    hold on
 %loglog(1/mont_iter*sum(error_me_svrg_noise,1))
 %loglog(1/mont_iter*sum(error_me_sgd_noise,1))
    loglog(1/mont_iter*sum(error_laborde1,1))
    loglog(1/mont_iter*sum(error_sgd,1))
    loglog(1/mont_iter*sum(error_sgd2,1))
    loglog(1/mont_iter*sum(error_svrg,1))
% loglog(1/mont_iter*sum(std_rate,1))
%loglog(1/mont_iter*sum(error_me_sgd,1))
 loglog(error_GD)
% sum_bound_me = 1/mont_iter*sum(bound_me,1);
% loglog(sum_bound_me)
%sum_bound_laborde = 1/mont_iter*sum(bound_laborde,1);
%loglog(sum_bound_laborde)
%plot(sum_bound_laborde)