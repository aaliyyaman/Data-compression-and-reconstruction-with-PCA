clear;
clc;
close all

%% we use the principal components to reduce the feature dimension of our dataset 
%% By using the projected data, you can train your model faster as there are less dimensions in the input.
%% YANİ, KISACA PCA İLE M*N OLAN DATANI EN İYİ TEMSİL EDEN K TANE FEATUREYE ÇEVİRİYON BÖYLELİKLE
%% DATAN M*K OLUYOR, YANİ Z. Z=M*K


load ('ex7data1.mat');

[U,~,Xn]=pca(X);
K = 1;
Z = projectData(Xn, U, K);
Xappx= recoverData(Z, U, K); % recover an approximation of the data


figure
subplot(1,2,1)
plot(Xn(:, 1), Xn(:, 2), 'bo','MarkerSize',8);
title('ORIGINAL')
subplot(1,2,2)
plot(Xn(:, 1), Xn(:, 2), 'bo','MarkerSize',8);
hold on
plot(Xappx(:, 1), Xappx(:, 2), 'ro','MarkerSize',8);
hold on;
for i = 1:size(Xn, 1)
    drawLine(Xn(i,:), Xappx(i,:), 'k', 'LineWidth', 2);
end
title('Recovered')
hold off

%% PART 2:  PCA on face images
clear
close all
clc


load ('ex7faces.mat')
% a=randperm(5000);
figure
displayData(X(1:100,:)); % 32*32 olduğu için çok kötü

% X is 5000*1024, 5000 face images
% Each row of X corresponds to one face image (a row vector of length 1024)
% each  32*32 in grayscale.

[X_norm] = featureNormalize(X);
[U, ~] = pca(X_norm);
figure
displayData(U(:, 1:36)');

% project the face dataset onto only the first 100 principal components
K = 100;
Z = projectData(X_norm, U, K);
fprintf('The projected data Z has a size of: %d x %d \n', size(Z));

% PCA can help speed up your learning algorithm signicantly. For example, 
% if you were training a neural network to perform person recognition (gven a 
% face image, predict the identitfy of the person), you can use the dimension 
% reduced input of only a 100 dimensions instead of the original pixels.

X_rec  = recoverData(Z, U, K);
% Display normalized data
figure
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;

%% PART 3:  PCA on my photo
clear
close all
clc

a=imread('biber.jpg');
b=a(:,:,2);
X=double(b)/255;
% imshow(X)=imshow(b) aynıdır
% imagesc(X)= imagesc(b) renkli
[U,S,Xn]=pca(X);


% K = 20; % or find K w,th below algorithm
ss=sum(sum(S));
for K=1:size(X,2)
    ss2(K)=S(K,K);
    if sum(ss2)/ss >=0.99
        break
    end
end

Z = projectData(Xn, U, K);
Xappx= recoverData(Z, U, K);


figure
subplot(1, 2, 1);
imshow(b)
title(sprintf('Original: %d features', size(X,2)));
axis square;

subplot(1, 2, 2);
imshow(Xappx)
title(sprintf('Recovered: with top %d principal component', K));
axis square;

function [U, S, X] = pca(X)
X1=featureNormalize(X);
% U = zeros(n);
% S = zeros(n);
m=size(X,1);
sigma = (1/m)*(X'*X);
[U, S , ~] = svd(sigma);
end

function [X_norm] = featureNormalize(X)

mu = mean(X);
sigma = std(X);
X_norm =(X-mu)./sigma ;

end

function Z = projectData(X, U, K)
% Z = zeros(size(X, 1), K);
  U_reduce = U(:,(1:K));   % n x K
  Z = X * U_reduce;        % m x k
end

function X_rec = recoverData(Z, U, K)
% X_rec = zeros(size(Z, 1), size(U, 1));
%                    m     *   n
X_rec = Z * U(:,1:K)'; %=m*n

end


