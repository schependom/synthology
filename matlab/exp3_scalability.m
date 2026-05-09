% exp3_scalability.m
%
% Exp 3 reasoning-wall scalability: time and peak memory to generate
% budget-matched training data vs. number of OWL2Bench universities,
% for UDM OWL2Bench baseline (owl_full reasoner) and Synthology.
%
% Data source: results/exp3_scalability/timing_summary.csv
% Generate that file with: bash jobscripts/exp3-scalability-collect.sh
%
% CSV format: u,method,time_sec,time_min,mem_mb,status
% Failed / missing rows contain NaN — those bars are omitted from the plot.

disp('IMPORTANT: delete exp3_scalability.pdf before running this script!');

FS = 24;
style = common(FS);
C = style.C;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(repoRoot, 'paper', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ── Load CSV ─────────────────────────────────────────────────────────────────
csvPath = fullfile(repoRoot, 'results', 'exp3_scalability', 'timing_summary.csv');

if ~isfile(csvPath)
    error('timing_summary.csv not found at %s.\nRun: bash jobscripts/exp3-scalability-collect.sh', csvPath);
end

T = readtable(csvPath, 'TextType', 'string');

% Collect unique u values (sorted)
u_all = unique(str2double(string(T.u)));
u_all = sort(u_all(~isnan(u_all)));
n_u   = numel(u_all);

time_baseline   = NaN(n_u, 1);
time_synthology = NaN(n_u, 1);
mem_baseline    = NaN(n_u, 1);
mem_synthology  = NaN(n_u, 1);

bl_failed = false(n_u, 1);  % true = ran until failure (hatch bar)

for i = 1:n_u
    u_val = u_all(i);

    bl_row = T(str2double(string(T.u)) == u_val & strcmpi(string(T.method), 'baseline'), :);
    sb_row = T(str2double(string(T.u)) == u_val & strcmpi(string(T.method), 'synthology'), :);

    if ~isempty(bl_row)
        v = str2double(string(bl_row.time_min(1)));
        m = str2double(string(bl_row.mem_mb(1)));
        if ~isnan(v)
            time_baseline(i) = v;
            mem_baseline(i)  = m / 1024;
            bl_failed(i) = ~strcmpi(string(bl_row.status(1)), 'OK');
        end
    end

    if ~isempty(sb_row)
        v = str2double(string(sb_row.time_min(1)));
        m = str2double(string(sb_row.mem_mb(1)));
        if ~isnan(v)
            time_synthology(i) = v;
            mem_synthology(i)  = m / 1024;
        end
    end
end

labels  = {'UDM Baseline', 'Synthology'};
xticks  = 1:n_u;
xlabels = arrayfun(@(u) sprintf('%d', u), u_all, 'UniformOutput', false);

fig = figure('Position', [100, 100, 1400, 560], 'Color', 'w');

% ── Left subplot: wall-clock time ────────────────────────────────────────────
ax1 = subplot(1, 2, 1);

time_data = [time_baseline, time_synthology];
b1 = bar(xticks, time_data, 'grouped');
b1(1).FaceColor = C.KULijsblauw;
b1(2).FaceColor = C.KULcorporate;
b1(1).FaceAlpha = 0.88;
b1(2).FaceAlpha = 0.88;

set(ax1, 'YScale', 'log');
set(ax1, 'XTick', xticks, 'XTickLabel', xlabels, ...
    'FontSize', FS, 'TickLabelInterpreter', 'latex');

xlabel('Number of universities',               'FontSize', FS, 'FontWeight', 'bold');
ylabel('Generation time (minutes, log scale)', 'FontSize', FS, 'FontWeight', 'bold');
title('Wall-clock Time',                       'FontSize', FS + 2, 'FontWeight', 'bold');

legend(labels, 'Location', 'northwest', 'FontSize', FS - 2, 'Interpreter', 'latex');
box off;
grid on;
set(ax1, 'GridLineStyle', ':', 'GridAlpha', 0.5);

% ── Right subplot: peak memory ───────────────────────────────────────────────
ax2 = subplot(1, 2, 2);

mem_data = [mem_baseline, mem_synthology];
b2 = bar(xticks, mem_data, 'grouped');
b2(1).FaceColor = C.KULijsblauw;
b2(2).FaceColor = C.KULcorporate;
b2(1).FaceAlpha = 0.88;
b2(2).FaceAlpha = 0.88;

set(ax2, 'YScale', 'log');
set(ax2, 'XTick', xticks, 'XTickLabel', xlabels, ...
    'FontSize', FS, 'TickLabelInterpreter', 'latex');

xlabel('Number of universities',      'FontSize', FS, 'FontWeight', 'bold');
ylabel('Peak memory (GB, log scale)', 'FontSize', FS, 'FontWeight', 'bold');
title('Peak Memory',                  'FontSize', FS + 2, 'FontWeight', 'bold');

legend(labels, 'Location', 'northwest', 'FontSize', FS - 2, 'Interpreter', 'latex');
box off;
grid on;
set(ax2, 'GridLineStyle', ':', 'GridAlpha', 0.5);

% ── Export ───────────────────────────────────────────────────────────────────
outFile = fullfile(outDir, 'exp3_scalability.pdf');
if isfile(outFile)
    delete(outFile);
end
exportgraphics(fig, outFile, 'ContentType', 'vector');
fprintf('Saved: %s\n', outFile);
close(fig);
