% Image descriptor based on Histogram of Orientated Gradients (HOG) for
% gray-level images. This code was developed for the work: 
% O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, "Trainable Classifier-Fusion 
% Schemes: An Application To Pedestrian Detection," in 12th Int. IEEE Conf. 
% Intelligent Transportation Systems, 2009, St. Louis, 2009. vol.1, pp. 432-437. 
% In case of publication with this code, please cite the paper above.
% Major revision was done by W.-S. Lu at University of Victoria.
% Last modified: June 15, 2020.
% Input:
% Im: input image. 
% d: d x d is the size of basic image block to perform HOG.
% Typically d is in the range from 3 to 7.
% B: number of histogram bins. Typically B is set to 7 or 9. 
% Output:
% h: HOG feature vector of input image. 
function h = hog20(Im,d,B)
t = floor(d/2); % Stride t=7(no overlap) t=1 (large overlap)
[N,M] = size(Im);
k1 = (M-d)/t;
c1 = ceil(k1);
k2 = (N-d)/t;
c2 = ceil(k2);
if c1 - k1 > 0
   M1 = d + t*c1;
   Im = [Im fliplr(Im(:,(2*M-M1+1):M))];
end
if c2 - k2 > 0
   N1 = d + t*c2;
   Im = [Im; flipud(Im((2*N-N1+1):N,:))];
end
[N,M] = size(Im);
nx1 = 1:t:M-d+1;
nx2 = d:t:M;
ny1 = 1:t:N-d+1;
ny2 = d:t:N;
Lx = length(nx1);
Ly = length(ny1);
hz = Lx*Ly*B;
H = zeros(hz,1); 
Im = double(Im);
k = 1;
hx = [-1 0 1];
% hx = [-1 0 1; -2 0 2; -1 0 1]; % Sobel operator in horizontal direction 
hy = -hx';                       
grad_xr = imfilter(Im,hx);       % gradient in horizontal direction 
grad_yu = imfilter(Im,hy);       % gradient in vertical direction 
magnit = sqrt(((grad_yu.^2) + (grad_xr.^2))); 
angles = atan2(grad_yu, grad_xr);
for m = 1:Lx
    for n = 1:Ly
        angles2 = angles(ny1(n):ny2(n),nx1(m):nx2(m));
        magnit2 = magnit(ny1(n):ny2(n),nx1(m):nx2(m));
        v_angles = angles2(:);    
        v_magnit = magnit2(:);
        K = length(v_angles);
        % assembling the histogram with B bins (range of 360/B degrees per bin)
        bin = 1;
        h2 = zeros(B,1);
        for ang_lim = -pi+2*pi/B:2*pi/B:pi
            for i = 1:K
                if v_angles(i) < ang_lim
                    v_angles(i) = 100;
                    h2(bin) = h2(bin) + v_magnit(i);
                end
            end
            bin = bin + 1;
        end
        h2 = h2/(norm(h2)+0.01);
        h((k-1)*B+1:k*B,1) = h2;
        k = k + 1;
    end
end