function exportWavefunctions(qqbarPath, outDir, opts)
% exportWavefunctions(qqbarPath, outDir, opts)
%
% Exports the LO BOEFT radial wave functions computed with the QQbarSpectrum
% code (read-only: the QQbarSpectrum folder is only added to the path, no
% file there is modified) to MATLAB v7 .mat files that the Python
% phenomenology code (../python) reads.
%
% For every family the potential handle is taken from the QQbarSpectrum
% file itself ('potential' option) and solved with its own
% solveRadialSchrodinger, so the exported functions are exactly those of
% QQbarSpectrum. Only the number of returned states is changed (nstates).
%
% INPUT  qqbarPath : path to the QQbarSpectrum folder
%        outDir    : folder where the .mat files are written
%        opts      : (optional) struct with fields
%                    .N (mesh points, default 100), .rmax (default 30 GeV^-1),
%                    .nstates (default 12), .mb (bottom mass, default 4.88)
%
% OUTPUT files: wf_c_nomix.mat, wf_b_nomix.mat
%   Each contains the mesh (r, h, lambda, N, rmax, m) and, for each family F, the
%   fields  F_E (1 x nstates energies, GeV), F_C (1 x N x nstates)
%   Lagrange-mesh coefficients c_i of every state, normalized as
%   sum_i c^2 = 1, so that  int_0^inf f(r) u_a(r) u_b(r) dr
%   = sum_i c_a,i c_b,i f(r_i)  (Gauss-Lagrange quadrature) and
%   u(r_i) = c_i / sqrt(h*lambda_i).
%
% Families (no quarkonium-hybrid mixing, mix = false):
%   Ql<l>    : quarkonium, orbital l = 0,1,2,3, from QQbarS0J<l> (S = 0, J = l;
%              at LO the radial functions do not depend on S and J)
%   Hj<J>    : hybrids with J = l (x) 1_gluon = 0,1,2 and S = 0 (Jcal = J), from GQQbarS0Jcal<J>;
%              channels (see their potentialMatrix): Hj0 [l=1], Hj1 [l=1; l=2; l=0],
%              Hj2 [l=2; l=3; l=1]. The S = 1 hybrids have the same radial functions at LO
%              and are built in Python (emlib/states.py). (v8 and older exports: Hjc0..Hjc2.)
%
% Author: EM transitions phenomenology (Sandra Tomas), 2026 (v10: hybrid families restored)

% Creates a mesh for the schrodinger solver
if nargin < 3, opts = struct(); end
if ~isfield(opts,'N'),       opts.N = 100;      end
if ~isfield(opts,'rmax'),    opts.rmax = 30;    end
if ~isfield(opts,'nstates'), opts.nstates = 12; end
if ~isfield(opts,'mb'),      opts.mb = 4.88;    end

addpath(qqbarPath);
if ~exist(outDir,'dir'), mkdir(outDir); end
sopts.N = opts.N; sopts.rmax = opts.rmax;
mesh = radialMesh(opts.N, opts.rmax);

% For each charmonium and bottomonium
masses = {'c', 1.496; 'b', opts.mb};
for f = 1:size(masses,1)
    flav = masses{f,1}; m = masses{f,2};
    % We reset the config to avoid any previous settings
    quarkoniumConfig('reset'); quarkoniumConfig('mix', false);
    hybridConfig('reset');     hybridConfig('mix', false);
    % S structure gives the mesh parameters, and "r" intervals of the wavefunctions
    S = struct();
    S.r = mesh.r; S.h = mesh.h; S.lambda = mesh.lambda; S.m = m;
    S.N = opts.N; S.rmax = opts.rmax;
    % Families: quarkonium l = 0,1,2,3 (the radial functions are those of S = 0, J = l)
    % and S = 0 hybrids J = 0,1,2
    fams = {'Ql0', @QQbarS0J0; 'Ql1', @QQbarS0J1; 'Ql2', @QQbarS0J2; 'Ql3', @QQbarS0J3; ...
            'Hj0', @GQQbarS0Jcal0; 'Hj1', @GQQbarS0Jcal1; 'Hj2', @GQQbarS0Jcal2};
    % For each family we get the potential matrix from the corresponding code in QQbarSpectrum
    % and then solve the radial schrodinger equation with the defined mesh
    for k = 1:size(fams,1)
        [~,~,~,inf0] = fams{k,2}(m, 'potential');
        [E, C] = solveFamily(inf0.V, m, opts.nstates, sopts);
        % Add to the S structure the energies and the coefficients of the wavefunctions for each family
        S.([fams{k,1} '_E']) = E; S.([fams{k,1} '_C']) = C;
    end
    save('-v7', fullfile(outDir, sprintf('wf_%s_nomix.mat', flav)), '-struct', 'S');
    fprintf('exported %s (m = %.3f), mix = false\n', flav, m);
end

quarkoniumConfig('reset'); hybridConfig('reset');
end


function [E, C3] = solveFamily(V, m, nst, sopts)
% Solve with the QQbarSpectrum solver and return the mesh coefficients as
% an (nch x N x nst) array (channel index runs fastest in info.C).
[E, ~, ~, info] = solveRadialSchrodinger(V, m, nst, sopts);
n = info.n; N = info.mesh.N;
C3 = zeros(n, N, numel(E));
for s = 1:numel(E)
    Cs = reshape(info.C(:,s), n, N);
    % fix the global sign: largest coefficient positive
    [~, imax] = max(abs(Cs(:)));
    C3(:,:,s) = sign(Cs(imax)) * Cs;
end
end
