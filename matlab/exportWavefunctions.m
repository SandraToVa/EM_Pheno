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
% OUTPUT files: wf_c_nomix.mat, wf_b_nomix.mat, wf_c_mix.mat
%   Each contains the mesh (r, h, lambda) and, for each family F, the
%   fields  F_E (1 x nstates energies, GeV), F_C (nch x N x nstates)
%   Lagrange-mesh coefficients c_{ch,i} of every state, normalized as
%   sum_{ch,i} c^2 = 1, so that  int_0^inf f(r) u_a(r) u_b(r) dr
%   = sum_i c_a,i c_b,i f(r_i)  (Gauss-Lagrange quadrature) and
%   u(r_i) = c_i / sqrt(h*lambda_i).
%   dV (1 x N): static gap V_Pi_u(r) - V_Sigma_g(r) at the mesh points (GeV).
%
% Channel content of each family (see the potentialMatrix of each file):
%   mix = false
%     Ql<l>    : quarkonium, orbital l = 0,1,2,3 from QQbarS0J<l>
%     Hjc0     : hybrid Jcal = 0, 1 channel  [l=1, Sigma_u only]       (p0)
%     Hjc1     : hybrid Jcal = 1, 3 channels [l=1 Pi (p1); l=2; l=0]   (s/d)1
%     Hjc2     : hybrid Jcal = 2, 3 channels [l=2 Pi (d2); l=3; l=1]   (p/f)2
%   mix = true (charm only, cF = 0 for other masses)
%     MixJ1    : GQQbarS0Jcal1 = QQbarS1J1, 6 channels
%                [Q(l=1,S=1); H(l=1,S=0); Q(l=2,S=1); Q(l=0,S=1); H(l=2,S=0); H(l=0,S=0)]
%     MixJ0    : QQbarS1J0, 2 channels [Q(l=1,S=1); H(l=1,S=0) Sigma]
%     MixJ2    : QQbarS1J2, 6 channels [Q(l=2,S=1); H(l=2,S=0); Q(l=3,S=1); Q(l=1,S=1); H(l=3,S=0); H(l=1,S=0)]
%     MixS0J0  : QQbarS0J0, 3 channels [Q(l=0,S=0); H(S=1) ...]  (only ch 1 used)
%     MixS0J1  : QQbarS0J1, 5 channels [Q(l=1,S=0); H(S=1) ...]  (only ch 1 used)
%
% Author: EM transitions phenomenology (Sandra Tomas), 2026

if nargin < 3, opts = struct(); end
if ~isfield(opts,'N'),       opts.N = 100;      end
if ~isfield(opts,'rmax'),    opts.rmax = 30;    end
if ~isfield(opts,'nstates'), opts.nstates = 12; end
if ~isfield(opts,'mb'),      opts.mb = 4.88;    end

addpath(qqbarPath);
if ~exist(outDir,'dir'), mkdir(outDir); end
sopts.N = opts.N; sopts.rmax = opts.rmax;
mesh = radialMesh(opts.N, opts.rmax);

masses = {'c', 1.496; 'b', opts.mb};
for f = 1:size(masses,1)
    flav = masses{f,1}; m = masses{f,2};
    quarkoniumConfig('reset'); quarkoniumConfig('mix', false);
    hybridConfig('reset');     hybridConfig('mix', false);
    S = struct();
    S.r = mesh.r; S.h = mesh.h; S.lambda = mesh.lambda; S.m = m;
    S.N = opts.N; S.rmax = opts.rmax;
    % quarkonium l = 0,1,2,3 and hybrids Jcal = 0,1,2
    fams = {'Ql0', @QQbarS0J0; 'Ql1', @QQbarS0J1; 'Ql2', @QQbarS0J2; 'Ql3', @QQbarS0J3; ...
            'Hjc0', @GQQbarS0Jcal0; 'Hjc1', @GQQbarS0Jcal1; 'Hjc2', @GQQbarS0Jcal2};
    for k = 1:size(fams,1)
        [~,~,~,inf0] = fams{k,2}(m, 'potential');
        [E, C] = solveFamily(inf0.V, m, opts.nstates, sopts);
        S.([fams{k,1} '_E']) = E; S.([fams{k,1} '_C']) = C;
    end
    % static gap Pi_u - Sigma_g^+ on the mesh (GeV): (V_Pi - V_Sigma) from
    % the l = 1 Pi_u channel of GQQbarS0Jcal1 and the l = 1 quarkonium
    % potential (same centrifugal term, same constant shifts)
    [~,~,~,infH] = GQQbarS0Jcal1(m, 'potential');
    [~,~,~,infQ] = QQbarS0J1(m, 'potential');
    VH = infH.V(mesh.r'); VQ = infQ.V(mesh.r');
    S.dV = squeeze(VH(1,1,:) - VQ(1,1,:))' / m;
    save('-v7', fullfile(outDir, sprintf('wf_%s_nomix.mat', flav)), '-struct', 'S');
    fprintf('exported %s (m = %.3f), mix = false\n', flav, m);
end

% mixing (charm only)
m = 1.496;
quarkoniumConfig('reset'); quarkoniumConfig('mix', true);
hybridConfig('reset');     hybridConfig('mix', true);
S = struct(); S.r = mesh.r; S.h = mesh.h; S.lambda = mesh.lambda; S.m = m;
S.N = opts.N; S.rmax = opts.rmax;
fams = {'MixJ1', @QQbarS1J1; 'MixJ0', @QQbarS1J0; 'MixJ2', @QQbarS1J2; ...
        'MixS0J0', @QQbarS0J0; 'MixS0J1', @QQbarS0J1};
for k = 1:size(fams,1)
    [~,~,~,inf0] = fams{k,2}(m, 'potential');
    [E, C] = solveFamily(inf0.V, m, opts.nstates, sopts);
    S.([fams{k,1} '_E']) = E; S.([fams{k,1} '_C']) = C;
end
save('-v7', fullfile(outDir, 'wf_c_mix.mat'), '-struct', 'S');
fprintf('exported c, mix = true\n');
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
