clear
clc
load('X1600.mat')
load('Lte28.mat')
load('Te28.mat')
u = ones(1,1600);
ytr = [u 2*u 3*u 4*u 5*u 6*u 7*u 8*u 9*u 10*u];
Dtr = [X1600; ytr]; % Original training samples
ytest = 1 + Lte28(:)'; % Turn MNIST labels to the labels used in this esperiment
Dte = [Te28; ytest]; % Original testing samples
[N,P_tr] = size(X1600);
P_te = size(Te28,2);
L = 10;
K = 23;

% PCA on original dataset
[CK, meanK] = PCA_training(Dtr,N,P_tr,L,K); % Train classifier using PCA and Original Dataset
tt = cputime; % Used to measure speed of recognition
ind_pre = PCA_classifying(Te28,CK,meanK,L,P_te); % Classify Testing Dataset
tt = cputime - tt;
speed_of_recognition = floor(inv(tt/P_te)); % Speed of recognition in samples per second
[C,accuracy] = confusion_matrix_and_accuracy(ind_pre,ytest,L,P_te); % Confusion matrix and accuracy

% PCA on HOG of dataset
Dhtr = pre_processing_HOG(Dtr,N,P_tr); % Obtaining the HOG of training data
N_hog = size(Dhtr,1) - 1;
[CK_hog, meanK_hog] = PCA_training(Dhtr,N_hog,P_tr,L,K); % Train classifier using PCA and HOG of Dataset
tt_hog = cputime; % Used to measure speed of recognition
Dhte = pre_processing_HOG(Dte,N,P_te); % Obtaining the HOG of testing data
ind_pre_hog = PCA_classifying(Dhte(1:N_hog,:),CK_hog,meanK_hog,L,P_te); % Classify Testing Dataset
tt_hog = cputime - tt_hog;
speed_of_recognition_hog = floor(inv(tt_hog/P_te)); % Speed of recognition in samples per second
[C_hog,accuracy_hog] = confusion_matrix_and_accuracy(ind_pre_hog,ytest,L,P_te); % Confusion matrix and accuracy