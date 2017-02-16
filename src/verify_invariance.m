% MATLAB script to symbolically verify the transformation group given 
% in the Dubins vehicle example 

clear all
close all
clc

syms x1 x2 x3 x4 x5 x6 u1 u2 u3 u4 w1 w2 w3 w4 g1 g2 g3 L

x = [x1; x2; x3; x4; x5; x6];
u = [u1; u2; u3; u4];
w = [w1; w2; w3; w4];
gamma = [g1; g2; g3];

f = @(x, u, w) x + [u(1)*cos(x(3)) + w(1); 
                    u(1)*sin(x(3)) + w(2);
                    1/L*u(1)*tan(u(2));
                    u(3)*cos(x(6)) + w(3); 
                    u(3)*sin(x(6)) + w(4);
                    1/L*u(3)*tan(u(4))
                    ]; 

R = @(p) [cos(p)  -sin(p) 0    0       0     0;
          sin(p)   cos(p) 0    0       0     0;
            0        0    1    0       0     0; 
            0        0    0  cos(p)  -sin(p) 0;
            0        0    0  sin(p)   cos(p) 0;
            0        0    0    0       0     1];
        
                
phi     = @(x, gamma)  R(gamma(3))*x + [gamma; gamma];
phi_inv = @(x, gamma) R(-gamma(3))*( x - [gamma; gamma]); 

psi = @(w, gamma) [cos(gamma(3))  -sin(gamma(3))  0  0;
                   sin(gamma(3))   cos(gamma(3))  0  0;
                       0                0        cos(gamma(3))  -sin(gamma(3));
                       0                0        sin(gamma(3))   cos(gamma(3))]*w;

% should come out idential to f(x, u, w) 
transformed_dynamics = simplify(phi_inv(f(phi(x, gamma), u, psi(w, gamma)), gamma))


