% Using the Principal Components acquired from PCA_training function
% to classify a number of test samples
function ind_pre = PCA_classifying(Xte,CK,meanK,L,P_te)
E_jp = zeros(L,P_te);
for j=1:L
    CK_j = CK(:,:,j);
    meanK_j = meanK(:,j);
    Z_j = CK_j' * (Xte - meanK_j); % Compute the projection
    Xte_est = CK_j*Z_j + meanK_j; % Approximation of test data
    % Euclidean distance between actual and approximate test data
    for p=1:P_te
        E_jp(j,p) = norm(Xte(:,p) - Xte_est(:,p));
    end
end

ind_pre = zeros(1,P_te);
for p=1:P_te
    ind_pre(p) = find( E_jp(:,p) == min(E_jp(:,p)) );
end
end