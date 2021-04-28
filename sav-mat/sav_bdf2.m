function [phis, rs] = sav_bdf2()
global dt epsilon k2 k4 C0 h N beta epsilon
N = 64;
T = 1000;
t = 0;
dt = 1e-5;
h = 1/N;
beta = 2;
C0 = 0;
epsilon = 1e-2;
x = h*(0:N-1);
y = h*(0:N-1);

[xx,yy] = meshgrid(x,y);
phi0 = phi_init(xx,yy);

phis = zeros(T,N^2);
rs = zeros(T, N^2);

[k_x,k_y,kx,ky,kxx,kyy,k2,k4] = prepare_fft2(N);

r0 = fun_r_init(phi0);

[phi1, r1] = sav_centre();
phis(1,:)=phi0(:);
phis(2,:)=phi1(:);
rs(1,:)=r0(:);
rs(2,:)=r1(:);

for nt = 3:T
    t = t+dt;
    phi_star = 2*phi1 - phi0;
    
    % step 1
    H = fun_H(phi_star);
    
    g = get_rhs(phi1,phi0,r1,r0,H);  
    
    psiA = inv_A(H);
    psiB = inv_A(g);  
        
    gamma = fft2(H.*psiA);
    gamma = gamma(1,1)*h*h;
    
    % Step 2      
    Hphi = fft2(H.*psiB);
    Hphi = Hphi(1,1)*h*h/(1+dt*gamma/3);
    
    % Step 3
    phi = psiB - dt/3*Hphi.*psiA ;  
    phis(nt,:) = phi(:);
%% update phi0  
    rold = r1;
    r1 = fun_r(phi,phi1,phi0,r1,r0,H);
    r0 = rold;  
    
    phi0 = phi1;      
    phi1 = phi;     
    
end

for t=1:round(T/30):T
    subplot(111)
    imagesc(reshape(phis(t,:),N,N));
%     subplot(122)
%     scatter(t,compute_energy(phis(t,:),rs(t,:)));
    pause(0.2);
end
energym = zeros(T,1);
energyr = zeros(T,1);
for t=1:T
[energym(t), energyr(t)]=compute_energy(reshape(phis(t,:),N,N),reshape(rs(t,:),N,N));
end
subplot(121)
loglog(1:T,energym);
subplot(122)
loglog(1:T,energyr);
% close;

end

function r = fun_r_init(phi)
global C0 h
E1 = fft2(F(phi));
r  = sqrt(E1(1,1)*h*h + C0);
end

function r = fun_r(phi,phi1,phi0,r1,r0,H)
global h
Hphi0 = fft2(H.*(4*phi1-phi0)/3);
Hphi0 = Hphi0(1,1)*h*h;
Hphi1 = fft2(H.*phi);
Hphi1 = Hphi1(1,1)*h*h;
g1 = (4*r1-r0)/3 - 1/2*Hphi0;
r = 1/2*Hphi1 + g1;
end

function r = fun_H(phi)
global C0 h
E1 = fft2(F(phi));
r = F_derivative(phi)./sqrt(E1(1,1)*h*h+C0);
end

function r = get_rhs(phi1,phi0,r1,r0,H)
global dt h
Hphi0 = fft2(H.*(4*phi1-phi0));
Hphi0 = Hphi0(1,1)*h*h;
r = (4*phi1-phi0)/3 - 2/9*dt*(4*r1-r0 - 1/2*Hphi0).*H;
end

function lap=lap_diff(phi)
global k2
lap=real(ifft2((k2.*fft2(phi))));
end

function r = inv_A(phi)
global dt k2 
    r = real(ifft2(fft2(phi)./(1-2/3*dt*k2)));
end


function r = F_derivative(phi)
global beta epsilon
    r = phi.*(phi.^2-beta-1)/epsilon^2;
end

function r = F(phi)
global beta epsilon
    r = (phi.^2-beta-1).^2/(4*epsilon^2);
end

function r = phi_init(xx,yy)
global N
r = 5e-2*sin(2*pi*xx).*sin(2*pi*yy);
% r = cos(2*pi*xx).*cos(pi*yy);
end

function [k_x,k_y,kx,ky,kxx,kyy,k2,k4] = prepare_fft2(N)
k_x = 1i*[0:N/2 -N/2+1:-1]*(2*pi);
k_y = 1i*[0:N/2 -N/2+1:-1]*(2*pi);

[kx,  ky ] = meshgrid(k_x,k_y);
k2x = k_x.^2;
k2y = k_y.^2;
[kxx, kyy] = meshgrid(k2x,k2y);
k2 = kxx + kyy;
k4 = k2.^2;
end

function [e1,e2] = compute_energy(phi,r)
global h
lphi = lap_diff(phi);
philphi = sum(sum(phi.*lphi));
% philphi = fft2(phi.*lphi);
% philphi = philphi(1,1)*h*h;
% r2 = fft2(r.*r);
% r2 = r2(1,1)*h*h;
r2 = sum(sum(r.*r));
e1 = philphi + r2;
% f = fft2(F(phi));
% f = f(1,1)*h*h;
f = sum(sum(F(phi)));
e2 = philphi/2+f;
end

