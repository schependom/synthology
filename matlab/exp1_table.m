% --- Experiment 1: Print LaTeX Table ---
C = kulcolors(); % Shared palette for any plots added to this script.

methods = {'Random Corruption', 'Constrained Random', 'Proof-Based'};

% Dummy Data: [PR-AUC, F1-Score, FPR]
% Replace with your actual experimental results
results = [
    0.65, 0.62, 0.28;  
    0.72, 0.70, 0.18;  
    0.88, 0.85, 0.05   
];

fprintf('\\begin{table}[H]\n');
fprintf('  \\centering\n');
fprintf('  \\begin{tabular}{l c c c}\n');
fprintf('    \\toprule\n');
fprintf('    \\textbf{Negative Strategy} & \\textbf{PR-AUC ($\\uparrow$)} & \\textbf{F1-Score ($\\uparrow$)} & \\textbf{FPR ($\\downarrow$)} \\\\\n');
fprintf('    \\midrule\n');

for i = 1:length(methods)
    fprintf('    %s & %.2f & %.2f & %.2f \\\\\n', methods{i}, results(i,1), results(i,2), results(i,3));
end

fprintf('    \\bottomrule\n');
fprintf('  \\end{tabular}\n');
fprintf('  \\caption{Experiment 1: Impact of negative sampling strategies on RRN performance.}\n');
fprintf('  \\label{tab:exp1_results}\n');
fprintf('\\end{table}\n');
