% bestx: the best solution found by your algorithm
% recordedAvgY: array of  average fitnesses of each generation
% recordedBestY: array of best fitnesses of each generation
function [bestx, recordedAvgY, recordedBestY]=EA(funcName,n,lb,ub,nbEvaluation)
warning on MATLAB:divideByZero
if nargin < 5
  error('input_example :  not enough input')
end
global objective;
eval(sprintf('objective=@%s;',funcName)); % Do not delete this line
% objective() is the evaluation function, for example : fitness = objective(x) 

%% Your algorithm
addpath('./utils/');
% n is the dimension of solution, lb&ub means low(up) bound
% nbEvaluation is the times of evaluations can be done during the
% algorithm

[bestx, recordedAvgY, recordedBestY] = FEP_FLB(n,lb,ub,nbEvaluation);
function [bestx, recordedAvgY, recordedBestY] = FEP_FLB(n,lb,ub,nbEvaluation)
population_size = 100;
tournament_size = 10;
y = zeros(1, population_size);
bestx = zeros(1, n);
recordedAvgY = [];
recordedBestY = [];
    
%parameter for self-adaptation
tao = 1 / sqrt(2*sqrt(population_size));
tao_ = 1 / sqrt(2*population_size);
lower_bound = 0.0001;
% init the population uniformly
x = zeros(population_size, 2*n);
x(1:population_size, 1:n) = lb + (ub - lb) .* rand(population_size, n);
x(1:population_size, n+1:2*n) =  3 .* ones(population_size, n);

% Evaluate the fitness score for each individual
for i = 1 : population_size
    y(i) = -1 * objective(x(i, 1:n));
    nbEvaluation = nbEvaluation - 1;
end
recordedAvgY(end+1) = mean(y(:));
[recordedBestY(end+1), index] = max(y(:));
bestx = x(index,1:n);

while true
    x_ = zeros(population_size, 2*n);    
    % mutation
    for i = 1 : population_size
        global_random = normrnd(0,1);
        for j = 1 : n
            j_random = normrnd(0, 1);
            % for any j component, generate an eta value.            
            x_(i, n+j) = x(i, n+j) * exp(tao_ * global_random + tao * j_random);
            x_(i, n+j) = boundData(x_(i, n+j), lower_bound);
            
            bias = x_(i, n+j) * cauchy(1);
            x_(i, j) = boundData(x(i, j) + bias, lb, ub);
        end
        
    end
    % evaluation
    y_ = zeros(1, population_size);
    for i = 1 : population_size
        y_(i) = -1 * objective(x_(i, 1:n));
        nbEvaluation = nbEvaluation - 1;        
    end
    % selection
    tournament_score = zeros(1, 2 * population_size);
    total_x = [x; x_];
    total_y = [y, y_];
    for i = 1 : 2 * population_size
        oppo = total_y(randperm(numel(total_y), tournament_size));
        tournament_score(i) = sum(oppo < total_y(i));
    end
    [~, score_index] = sort(tournament_score, "descend");
    
    x = total_x(score_index(1:population_size), :);
    y = total_y(score_index(1:population_size));
    
    recordedAvgY(end+1) = mean(y(:));
    [recordedBestY(end+1), index] = max(y(:));
   
    if nbEvaluation <= 0
            % exit
        bestx = x(index,1:n);
        return;
    end 
end
end

function [bestx, recordedAvgY, recordedBestY] = IFEP_Test(n,lb,ub,nbEvaluation)
population_size = 10;
tournament_size = 10;
y = zeros(1, population_size);
bestx = zeros(1, n);
recordedAvgY = [];
recordedBestY = [];
eta_bound2 = [];

    
%parameter for self-adaptation
tao = 1 / sqrt(2*sqrt(population_size));
tao_ = 1 / sqrt(2*population_size);
lower_bound = 0.001;
upper_bound = 6;
generation_interval = 25;
% init the population uniformly
x = zeros(population_size, 2*n);
x(1:population_size, 1:n) = lb + (ub - lb) .* rand(population_size, n);
x(1:population_size, n+1:2*n) = 3 .* ones(population_size, n);
eta = 3 .* ones(1, n);
etad = eta ./ 2;


first_gen = lb + (ub - lb) .* rand(2 * population_size, n);

% Evaluate the fitness score for each individual
for i = 1 : 2*population_size
    y(i) = -1 * objective(first_gen(i, 1:n));
    nbEvaluation = nbEvaluation - 1;
end

[~, score_index] = sort(y, "descend");
    
x(1:population_size, 1:n) = first_gen(score_index(1:population_size), 1:n);
y = y(score_index(1:population_size));

recordedAvgY(end+1) = mean(y(:));
[recordedBestY(end+1), index] = max(y(:));
bestx = x(index,1:n);

count = 1;
while true
    offspring = zeros(population_size, 2*n);
    y_ = zeros(1, population_size);
    x_c = zeros(1, 2*n);
    x_g = zeros(1, 2*n);
    
    % mutate standard deviations and x
    for i = 1 : population_size
        global_random = normrnd(0,1);
        for j = 1 : n
            j_random = normrnd(0, 1);
            
            x_c(n+j) = x(i, n+j) * exp(tao_ * global_random + tao * normrnd(0,1));            
            x_g(n+j) = x(i, n+j) * exp(tao_ * global_random + tao * j_random);
            
            x_c(j) = boundData(x(i, j) + x_c(n+j) * cauchy(1), lb, ub);
            x_g(j) = boundData(x(i, j) + x_g(n+j)*j_random, lb, ub);
        end
        % evaluation
        y_c = -1 * objective(x_c(1:n));
        y_g = -1 * objective(x_g(1:n));
        nbEvaluation = nbEvaluation - 2;        
        if y_c >= y_g
            y_(i) = y_c;            
            offspring(i, :) = x_c;
        else
            y_(i) = y_g;
            offspring(i, :) = x_g;
        end
        if nbEvaluation <= 0
            break;  
        end
    end
    % selection
    tournament_score = zeros(1, 2 * population_size);
    total_x = [x; offspring];
    total_y = [y, y_];
    for i = 1 : 2 * population_size
        oppo = total_y(randperm(numel(total_y), tournament_size));
        tournament_score(i) = sum(oppo < total_y(i));
    end
    [~, score_index] = sort(tournament_score, "descend");
    x = total_x([score_index(1:population_size)], :);
    y = total_y(score_index(1:population_size));

    mutation_bound = median(mean(x(1:population_size, n+1:2*n)));
    % dynamically control the update interval
    if count == generation_interval
        if mutation_bound >= upper_bound
            generation_interval = generation_interval + 10;
            x(1:population_size, n+1:2*n) = boundData(x(1:population_size, n+1:2*n), 0, mutation_bound);            
        else            
            x(1:population_size, n+1:2*n) = boundData(x(1:population_size, n+1:2*n), mutation_bound);
            generation_interval = boundData(ceil(generation_interval - 10), 5);
            
        end
        upper_bound = 0.875 * upper_bound + 0.125 * mutation_bound;
        count = 1;
    end    
    
    recordedAvgY(end+1) = mean(y(:));
    [recordedBestY(end+1), index] = max(y(:));
    if nbEvaluation <= 0 
        % exit
        bestx = x(index,1:n);
        return;
    end
    count = count + 1;    
end
end
    
end
