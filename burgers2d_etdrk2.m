% ETDRK2 for 2D Burgers' equation

% Parameters
N = 128;               % Number of grid points in each direction
L = 2 * pi;            % Domain size
x = linspace(0, L, N+1); x = x(1:end-1);
y = linspace(0, L, N+1); y = y(1:end-1);
[X, Y] = meshgrid(x, y);

nu = 0.01;             % Viscosity
dt = 0.01;             % Time step
t_end = 2;             % End time

% Wavenumber domain
k = [0:N/2-1 0 -N/2+1:-1]';  % Fourier wavenumbers
[KX, KY] = meshgrid(k, k);
K2 = KX.^2 + KY.^2;           % Laplacian in Fourier space

% Dealiasing filter
dealias = abs(KX) < 2/3 * (N/2) & abs(KY) < 2/3 * (N/2);

% Initial conditions
u0 = sin(X) .* cos(Y);
v0 = -cos(X) .* sin(Y);
U_hat = fft2(u0);
V_hat = fft2(v0);

% Exponential coefficients
L = -nu * K2;
E = exp(dt * L);
E2 = exp(dt * L / 2);

% Quadrature points for contour integral
M = 32;
r = exp(2i * pi * ((1:M) - 0.5) / M); % Quadrature roots of unity
LR = dt * L(:) + r(:).';

% Precompute ETDRK2 coefficients
f1 = dt * real(mean((exp(LR) - 1) ./ LR, 2));
f2 = dt * real(mean((exp(LR) - 1 - LR) ./ LR.^2, 2));

% Reshape coefficients
f1 = reshape(f1, N, N);
f2 = reshape(f2, N, N);

% Main time-stepping loop
t = 0;
plot_interval = 10;
while t < t_end
    % ETDRK2 stages
    [NLu_hat, NLv_hat] = nonlinear(U_hat, V_hat, KX, KY);

    au = E2 .* U_hat + f1 .* NLu_hat;
    av = E2 .* V_hat + f1 .* NLv_hat;

    [NLu_a_hat, NLv_a_hat] = nonlinear(au, av, KX, KY);

    U_hat = au + f2 .* (NLu_a_hat - NLu_hat);
    V_hat = av + f2 .* (NLv_a_hat - NLv_hat);

    % Time increment
    t = t + dt;

    % Visualization
    if mod(round(t / dt), plot_interval) == 0
        % Transform back to physical space
        u = real(ifft2(U_hat));
        v = real(ifft2(V_hat));

        % Plot the results
        subplot(1, 2, 1);
        contourf(x, y, u.', 50, 'LineColor', 'none'); 
        colorbar;
        title(['u(x, y, t = ', num2str(t), ')']);
        xlabel('x'); ylabel('y');

        subplot(1, 2, 2);
        contourf(x, y, v.', 50, 'LineColor', 'none'); 
        colorbar;
        title(['v(x, y, t = ', num2str(t), ')']);
        xlabel('x'); ylabel('y');

        % Update the figure
        drawnow;
    end

end

function [NLu_hat, NLv_hat] = nonlinear(U_hat, V_hat, KX, KY)
    u = real(ifft2(U_hat));
    v = real(ifft2(V_hat));
    ux = real(ifft2(1i * KX .* U_hat));
    uy = real(ifft2(1i * KY .* U_hat));
    vx = real(ifft2(1i * KX .* V_hat));
    vy = real(ifft2(1i * KY .* V_hat));

    NLu = -(u .* ux + v .* uy);
    NLv = -(u .* vx + v .* vy);
    
    NLu_hat = fft2(NLu);
    NLv_hat = fft2(NLv);
end
