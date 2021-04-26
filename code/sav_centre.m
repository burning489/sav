function [phis,rs] = sav_centre()
global dt epsilon k2 k4 C0 h N beta epsilon;
N = 128;
T = 20000;
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
r0 = fun_r_init(phi0);

[k_x,k_y,kx,ky,kxx,kyy,k2,k4] = prepare_fft2(N);

phis = zeros(T,N^2);
rs = zeros(T,N^2);
phis(1,:) = phi0(:);
rs(1,:) = r0(:);

for nt = 2:T
    t = t+dt;
    phi_star = phi0;    
    % step 1
    H = fun_H(phi_star);
    
    g = get_rhs(phi0,r0,H);
    
    psiA = inv_A(H);
    psiB = inv_A(g);    
        
    gamma = fft2(H.*psiA);
    gamma = gamma(1,1)*h*h;
    
    % Step 2      
    Hphi = fft2(H.*psiB);
    Hphi = Hphi(1,1)*h*h/(1+dt*gamma/2);
    
    % Step 3
    phi = psiB - dt/2*Hphi.*psiA ;     
   

    %% update phi0
    r0 = fun_r(phi,phi0,r0,H); 
    phi0 = phi;
    phis(nt,:) = phi(:);
    rs(nt,:) = r0(:);
    
end

% for t=1:round(T/50):T
%     subplot(111)
%     imagesc(reshape(phis(t,:),N,N));
%     pause(0.1);
% end

gif = figure;
filename = '../pics/phaseAnimation.gif';
for t = 1:round(T/50):T
    subplot(111)
    imagesc(reshape(phis(t,:),N,N));
    drawnow 
      % Capture the plot as an image 
      frame = getframe(gif); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if t == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end
close;
  
energym = zeros(T,1);
energyr = zeros(T,1);
for t=1:T
[energym(t), energyr(t)]=compute_energy(reshape(phis(t,:),N,N),reshape(rs(t,:),N,N));
end
subplot(121)
loglog(1:T,energym);
title('modified free energy')
xlabel('time')
subplot(122)
loglog(1:T,energyr);
title('raw free energy')
xlabel('time')
savefig('../pics/energy.fig')
saveas(gcf,'../pics/energy.jpg')
end

function r = phi_init(xx,yy)
r = 5e-2*sin(2*pi*xx).*sin(2*pi*yy);
% r = cos(2*pi*xx).*cos(pi*yy);
end

function r = fun_r_init(phi)
global C0 h
E1 = fft2(F(phi));
r  = sqrt(E1(1,1)*h + C0);
end

function r = fun_r(phi,phi0,r0,H)
global h
Hphi0 = fft2(H.*phi0);
Hphi0 = Hphi0(1,1)*h*h;
Hphi1 = fft2(H.*phi);
Hphi1 = Hphi1(1,1)*h*h;
g1 = r0 - 1/2*Hphi0;
r = 1/2*Hphi1+g1;
end

function r = fun_H(phi)
global C0 h
E1 = fft2(F(phi));
r = F_derivative(phi)./sqrt(E1(1,1)*h*h+C0);
end

function r = get_rhs(phi0,r0,H)
global dt h
Hphi0 = fft2(H.*phi0);
Hphi0 = Hphi0(1,1)*h*h;
r = phi0 - dt*r0.*H + dt/2*Hphi0.*H;
end

function lap=lap_diff(phi)
global k2
lap=real(ifft2((k2.*fft2(phi))));
end

function r = inv_A(phi)
global dt k2 
    r = real(ifft2(fft2(phi)./(1-dt*k2)));
end


function r = F_derivative(phi)
global beta epsilon
    r = phi.*(phi.^2-beta-1)/epsilon^2;
end

function r = F(phi)
global beta epsilon
    r = (phi.^2-beta-1).^2/(4*epsilon^2);
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
e1 = -1*philphi + r2;
% f = fft2(F(phi));
% f = f(1,1)*h*h;
f = sum(sum(F(phi)));
e2 = -1*philphi/2+f;
e1 = e1*h*h;
e2 = e2*h*h;
end

