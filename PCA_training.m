% Using Principal Component Analysis for Classification
function [CK, meanK] = PCA_training(Dtr,N,P_tr,L,K)
P_tr_j = P_tr / L;
for j=1:L
    Xtr_j = Dtr(1:N,((j-1)*P_tr_j+1):(j*P_tr_j)); % Get all samples belonging to each class
    mean_j = (mean(Xtr_j'))'; % Find the mean for all samples of each dimension 
    Xtr_j = Xtr_j - mean_j; % Make all samples mean-centered
    CV_j = (Xtr_j*Xtr_j') / P_tr_j; % Covariance of mean-centered samples of each dimension
    [CK_j,S] = eigs(CV_j,K); % Capture principal components
    CK(:,:,j) = CK_j;
    meanK(:,j) = mean_j;
end
end