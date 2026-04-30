% exp3_scalability.m
%
% Exp 3 reasoning-wall scalability: time and peak memory to generate
% budget-matched training data vs. number of OWL2Bench universities,
% for UDM OWL2Bench baseline (owl_full reasoner) and Synthology.
%
% Data extracted from LSF job logs (see jobscripts/exp3-generate-*.sh).
% Synthology budgets are matched to the baseline sample count at each scale.

disp('IMPORTANT: delete exp3_scalability.pdf before running this script!');

FS = 24;
style = common(FS);
C = style.C;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(repoRoot, 'paper', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ── Data ─────────────────────────────────────────────────────────────────────
% x-axis: number of OWL2Bench universities
universities = [5, 14, 20];

% Wall-clock generation time (minutes), median of available job logs
time_baseline   = [12.6,  1130.1, 186.7];
time_synthology = [74.4,    67.0,  65.7];

% Peak RSS memory (GB), median of available job logs
mem_baseline    = [14.8,   107.5,  57.5];
mem_synthology  = [ 4.6,     1.1,   1.1];
% ─────────────────────────────────────────────────────────────────────────────

labels  = {'UDM Baseline', 'Synthology'};
xticks  = 1:numel(universities);
xlabels = arrayfun(@(u) sprintf('%d', u), universities, 'UniformOutput', false);

fig = figure('Position', [100, 100, 1400, 560], 'Color', 'w');

% ── Left subplot: wall-clock time ────────────────────────────────────────────
ax1 = subplot(1, 2, 1);

time_data = [time_baseline(:), time_synthology(:)];
b1 = bar(xticks, time_data, 'grouped');
b1(1).FaceColor = C.KULijsblauw;
b1(2).FaceColor = C.KULcorporate;
b1(1).FaceAlpha = 0.88;
b1(2).FaceAlpha = 0.88;

set(ax1, 'YScale', 'log');
set(ax1, 'XTick', xticks, 'XTickLabel', xlabels, ...
    'FontSize', FS, 'TickLabelInterpreter', 'latex');

xlabel('Number of universities',              'FontSize', FS, 'FontWeight', 'bold');
ylabel('Generation time (minutes, log scale)', 'FontSize', FS, 'FontWeight', 'bold');
title('Wall-clock Time',                       'FontSize', FS + 2, 'FontWeight', 'bold');

legend(labels, 'Location', 'northwest', 'FontSize', FS - 2, 'Interpreter', 'latex');
box off;
grid on;
set(ax1, 'GridLineStyle', ':', 'GridAlpha', 0.5);

% ── Right subplot: peak memory ───────────────────────────────────────────────
ax2 = subplot(1, 2, 2);

mem_data = [mem_baseline(:), mem_synthology(:)];
b2 = bar(xticks, mem_data, 'grouped');
b2(1).FaceColor = C.KULijsblauw;
b2(2).FaceColor = C.KULcorporate;
b2(1).FaceAlpha = 0.88;
b2(2).FaceAlpha = 0.88;

set(ax2, 'YScale', 'log');
set(ax2, 'XTick', xticks, 'XTickLabel', xlabels, ...
    'FontSize', FS, 'TickLabelInterpreter', 'latex');

xlabel('Number of universities',        'FontSize', FS, 'FontWeight', 'bold');
ylabel('Peak memory (GB, log scale)',   'FontSize', FS, 'FontWeight', 'bold');
title('Peak Memory',                    'FontSize', FS + 2, 'FontWeight', 'bold');

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
