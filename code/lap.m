function lap = lap(n)
% laplacian matrix for periodic boundary condition
I = eye(n);
E = diag(ones(n-1,1),1);
D = E+E'-2*I;
lap = kron(D,I)+kron(I,D);
for i=1:n
    lap(i,(n-1)*n+i) = 1; % left
    lap((i-1)*n+1,i*n) = 1; % top
    lap((n-1)*n+i,i) = 1; % right
    lap(i*n,n*(i-1)+1) = 1; % bottom
end
end

