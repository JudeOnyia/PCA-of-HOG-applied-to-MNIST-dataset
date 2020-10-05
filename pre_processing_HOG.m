% HOG Training samples
function Dhtr = pre_processing_HOG(Dtr,N,P_tr)
Xtr = Dtr(1:N,:);
ytr = Dtr(N+1,:);
H = [];
for i = 1:P_tr
    xi = Xtr(:,i);
    mi = reshape(xi,28,28);
    hi = hog20(mi,4,9);
    H = [H hi];
end
Dhtr = [H; ytr];