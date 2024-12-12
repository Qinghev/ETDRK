% ETDRK2 for 2D Burgers' equation

% Parameters
N = 1280;               % Number of grid points in each direction
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
Q = dt * real(mean((exp(LR) - 1) ./ LR, 2));

% Reshape coefficients
Q = reshape(Q, N, N);

% Main time-stepping loop
t = 0;
plot_interval = 10;
while t < t_end
    % Nonlinear term
    u = real(ifft2(U_hat));
    v = real(ifft2(V_hat));
    ux = real(ifft2(1i * KX .* U_hat));
    uy = real(ifft2(1i * KY .* U_hat));
    vx = real(ifft2(1i * KX .* V_hat));
    vy = real(ifft2(1i * KY .* V_hat));

    NLu = -(u .* ux + v .* uy);
    NLv = -(u .* vx + v .* vy);

    NLu_hat = fft2(NLu) .* dealias;
    NLv_hat = fft2(NLv) .* dealias;

    % ETDRK2 stages
    a = E2 .* U_hat + Q .* NLu_hat;
    b = E2 .* V_hat + Q .* NLv_hat;

    ua = real(ifft2(a));
    va = real(ifft2(b));

    ux_a = real(ifft2(1i * KX .* a));
    uy_a = real(ifft2(1i * KY .* a));
    vx_a = real(ifft2(1i * KX .* b));
    vy_a = real(ifft2(1i * KY .* b));

    NLu_a = -(ua .* ux_a + va .* uy_a);
    NLv_a = -(ua .* vx_a + va .* vy_a);

    NLu_a_hat = fft2(NLu_a) .* dealias;
    NLv_a_hat = fft2(NLv_a) .* dealias;

    U_hat = E .* U_hat + Q .* NLu_a_hat;
    V_hat = E .* V_hat + Q .* NLv_a_hat;

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

    % Time increment
    t = t + dt;
end

% Transform back to physical space
u = real(ifft2(U_hat));
v = real(ifft2(V_hat));

% Visualization
figure;
subplot(1, 2, 1); contourf(x, y, u.', 50, 'LineColor', 'none'); colorbar; title('u(x,y,t)');
subplot(1, 2, 2); contourf(x, y, v.', 50, 'LineColor', 'none'); colorbar; title('v(x,y,t)');
