function phis = sav_bdf2()
global n T dt beta epsilon dx Delta phin phinm1 phim bn rn rnm1;
n = 32;
T = 100;
dt = 1e-5;
beta = 2;
C0 = 0;
epsilon = 1e-2;
dx = 1/n^2;
Delta = lap(n);
A = eye(n^2)-2/3*dt*Delta;
invA = inv(A);
phinm1 = init_phi();
phin = phinm1;
rnm1 = compute_rn(phinm1);
rn = rnm1;
phis = zeros(T,n^2);
phis(1,:)=phinm1;
phis(2,:)=phin;
for t = 3:T
    disp(t)
    phim = compute_phim();
    bn = compute_bn(phim);
    gn = compute_gn();
    gamman = sum(bn.*(A\gn)*dx);
    bnphinp1 = gamman/(1+dt/3*sum(bn.*A\bn*dx));
    phinp1 = A\gn - dt/3*A\bn*bnphinp1;
    rnp1 = compute_rn(phinp1);
    phinm1=phin; phin=phinp1;
    rnm1=rn; rn=rnp1;
    phis(t,:)=phin;
end
for t=2:10:T
%     figure(t)
    subplot(111)
    imagesc(reshape(phis(t,:),n,n));
    pause(0.5);
end
end

function phi0 = init_phi()
global n;
lx = linspace(0,1,n+1);
lx = lx(1:end-1);
[xx,yy] = meshgrid(lx,lx);
phi0 = 5e-2*sin(xx).*sin(yy);
phi0 = phi0(:);
end

function f = compute_f(phi)
global beta epsilon;
f = phi.*(phi.^2-beta-1)/epsilon^2;
end

function E1 = compute_E1(phim)
global dx beta epsilon;
E1 = sum((phim.^2-beta-1).^2/(4*epsilon^2)*dx);
end

function phim = compute_phim()
global n dt Delta beta epsilon phin;
phim = (eye(n^2)*(1+dt*beta/epsilon^2) - dt*Delta)\(phin+dt*compute_f(phin));
end

function bn = compute_bn(phim)
global C0;
bn = compute_f(phim)/sqrt(compute_E1(phim)+C0);
end

function gn = compute_gn()
global dx dt phin phinm1 rn rnm1 bn;
gn = 1/3*(4*phin-phinm1) - 2*dt/9*bn.*(4*rn-rnm1-1/2*sum(bn.*(4*phin-phinm1)*dx));
end

function rn = compute_rn(phi)
global C0;
rn = sqrt(compute_E1(phi)+C0);
end

