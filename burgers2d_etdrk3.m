% ETDRK3 for 2D Burgers' equation

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
r = exp(1i * pi * ((1:M) - 0.5) / M); % Quadrature roots of unity
LR = dt * L(:) + r(:).';

% Precompute ETDRK3 coefficients
Q = dt * real(mean((exp(LR/2) - 1) ./ LR, 2));
f1 = dt * real(mean((-4 - LR + exp(LR).*(4 - 3*LR + LR.^2)) ./ LR.^3, 2));
f2 = dt * real(mean((2 + LR + exp(LR).*(-2 + LR)) ./ LR.^3, 2));

% Reshape coefficients
Q = reshape(Q, N, N);
f1 = reshape(f1, N, N);
f2 = reshape(f2, N, N);

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

    % ETDRK3 stages
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

    c = E .* U_hat + f1 .* NLu_hat + f2 .* NLu_a_hat;
    d = E .* V_hat + f1 .* NLv_hat + f2 .* NLv_a_hat;

    % Update
    U_hat = c;
    V_hat = d;

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
