clear; clc; close all;
a = -1;b=-1;c=30;
f = @(x,y) a*x+b*y+c;
x_t = 1:10;
y_t = 1:10;
fall_point = [12,12,0];
z_t = f(x_t,y_t);
X = [x_t',y_t',z_t'];
X = [X;fall_point];
N = ndims(X);

X_ave = mean(X,1);
dx = bsxfun(@minus,X,X_ave);
C = (dx' * dx)/(N-1);
[R,D] = svd(C,0);
D = diag(D);
R2 = D(1)/sum(D);

x = dx * R(:,1);
x_min = min(x);
x_max = max(x);
difx = x_max - x_min;
Xa = (x_min - 0.05 * difx) * R(:,1)' + X_ave;
Xb = (x_max + 0.05 * difx) * R(:,1)' + X_ave;

end_x = [Xa;Xb];

figure('color','w')
axis equal
hold on;
plot3(end_x(:,1),end_x(:,2),end_x(:,3),'-r');
plot3(X(:,1),X(:,2),X(:,3),'-g');

% coefficient = polyfit([x_t,fall_point(1);y_t,fall_point(2)],[[z_t,0]],1)
% f_fit = @(x,y) coefficient(1)*x + coefficient(2)*y + coefficient(3);
% 
% x_y = [1:15;1:15];
% figure(1);
% hold on;
% plot(x_y,f(x_y(1,:),x_y(2,:)),'-g');
% plot(x_y,f_fit(x_y(1,:),x_y(2,:)),'-r');
