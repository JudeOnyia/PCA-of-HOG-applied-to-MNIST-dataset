% Compute confusion matrix and accuracy
function [C,accuracy] = confusion_matrix_and_accuracy(ind_pre,ytest,L,P_te)
% Confusion matrix
C = zeros(L,L);
for j = 1:L
    ind_j = find(ytest == j);
    for i = 1:L
        ind_pre_i = find(ind_pre == i);
        C(i,j) = length(intersect(ind_j,ind_pre_i));
    end
end
% Accuracy
accuracy = 0;
for i = 1:L
    accuracy = accuracy + C(i,i);
end
accuracy = (accuracy / P_te) * 100;
end