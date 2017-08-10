
%HT_table_array = cell(9,7);
% If you would like to fill in the HT_table_array quickly, uncomment the
% for loop below and the end statement at end, and comment out the graphing
% section down at line 228. Otherwise, comment out the for loop and end
% statement, and uncomment the graphing section
% patient = 9;
for patient=1:9
%% Task 0 - Loading data
% The following 9 lines load the appropriate patient data. Only one line
% should be active at a time 
% clc;
switch patient
    case 1
        load('1_a41178.mat'); % Patient 1
    case 2
        load('2_a42126.mat'); % Patient 2
    case 3
        load('3_a40076.mat'); % Patient 3
    case 4
        load('4_a40050.mat'); % Patient 4
    case 5
        load('5_a41287.mat'); % Patient 5
    case 6
        load('6_a41846.mat'); % Patient 6
    case 7
        load('7_a41846.mat'); % Patient 7
    case 8
        load('8_a42008.mat'); % Patient 8
    case 9
        load('9_a41846.mat'); % Patient 9
end

% We floor the data in order to estimate the probability distributions as
% mass functions
all_data = floor(all_data);

% These two equations figure out the appropriate indices for the data set
% to split it so that way 2/3's of the data is training, and 1/3 of the
% data is testing. The training_idx is floored to ensure that even with
% data sets that aren't divisible by 3, we can get something close
training_idx = floor(length(all_data) * 2/3);
testing_idx = length(all_labels);

training = all_data(:, 1:training_idx);
label_training = all_labels(1:training_idx);

testing = all_data(:, training_idx:testing_idx);
label_testing = all_labels(training_idx:testing_idx);

fprintf('Training data represents %f of data\n', training_idx/testing_idx);
fprintf('Testing data reprenets %f of data\n', (testing_idx - training_idx)/(testing_idx));
%% Task 1 - Enter the Likelihood Matrix
% Start off by calculating the prior probabilities, and setting up the H0
% and H1 training data
tabulation = tabulate(label_training);
training_H0 = training;
training_H1 = training;
P_H0 = tabulation(1,3)/100;
P_H1 = tabulation(2,3)/100;

% The idea is that we'll separate golden alarm data from non-golden alarm
% data. So long as this code is ran on MATLAB versions above R2017a, like
% the ones on Citrix
for i = 1:length(label_training)
    if(label_training(1,i) == 0)
        %%fprintf('%d equals 0\n', i);
        training_H1(1:7,i) = NaN;
    else
        %%fprintf('%d equals 1\n', i);
        training_H0(1:7,i) = NaN;
    end
end
training_H0 = rmmissing(training_H0,2);
training_H1 = rmmissing(training_H1,2);

% We figure out the minimum and maximmum values of data for later use
min_1 = min(training(1,:));
max_1 = max(training(1,:));
min_2 = min(training(2,:));
max_2 = max(training(2,:));
min_3 = min(training(3,:));
max_3 = max(training(3,:));
min_4 = min(training(4,:));
max_4 = max(training(4,:));
min_5 = min(training(5,:));
max_5 = max(training(5,:));
min_6 = min(training(6,:));
max_6 = max(training(6,:));
min_7 = min(training(7,:));
max_7 = max(training(7,:));

% We tabulate given H0 and H1 for our likelihood matrix
tab_1_H0 = tabulate(training_H0(1,:));
tab_1_H1 = tabulate(training_H1(1,:));
tab_2_H0 = tabulate(training_H0(2,:));
tab_2_H1 = tabulate(training_H1(2,:));
tab_3_H0 = tabulate(training_H0(3,:));
tab_3_H1 = tabulate(training_H1(3,:));
tab_4_H0 = tabulate(training_H0(4,:));
tab_4_H1 = tabulate(training_H1(4,:));
tab_5_H0 = tabulate(training_H0(5,:));
tab_5_H1 = tabulate(training_H1(5,:));
tab_6_H0 = tabulate(training_H0(6,:));
tab_6_H1 = tabulate(training_H1(6,:));
tab_7_H0 = tabulate(training_H0(7,:));
tab_7_H1 = tabulate(training_H1(7,:));

PMF_1 = zeros(3, range(training(1,:))+1);
for i = 1:length(PMF_1)
    PMF_1(1,i) = min_1+i-1;
    if(~isempty(find(tab_1_H0(:,1) == PMF_1(1,i))))
        PMF_1(3,i) = tab_1_H0(find(tab_1_H0(:,1) == PMF_1(1,i)), 3)/100;
    end
    if(~isempty(find(tab_1_H1(:,1) == PMF_1(1,i))))
        PMF_1(2,i) = tab_1_H1(find(tab_1_H1(:,1) == PMF_1(1,i)), 3)/100;
    end
end
for i = 1:length(PMF_1)
    if(PMF_1(3,i) == 0 && PMF_1(2,i) == 0)
        PMF_1(:,i) = [NaN; NaN; NaN];
    end
end
PMF_1 = rmmissing(PMF_1,2);

PMF_2 = zeros(3, range(training(2,:))+1);
for i = 1:length(PMF_2)
    PMF_2(1,i) = min_2+i-1;
    if(~isempty(find(tab_2_H0(:,1) == PMF_2(1,i))))
        PMF_2(3,i) = tab_2_H0(find(tab_2_H0(:,1) == PMF_2(1,i)), 3)/100;
    end
    if(~isempty(find(tab_2_H1(:,1) == PMF_2(1,i))))
        PMF_2(2,i) = tab_2_H1(find(tab_2_H1(:,1) == PMF_2(1,i)), 3)/100;
    end
end
for i = 1:length(PMF_2)
    if(PMF_2(3,i) == 0 && PMF_2(2,i) == 0)
        PMF_2(:,i) = [NaN; NaN; NaN];
    end
end
PMF_2 = rmmissing(PMF_2,2);

PMF_3 = zeros(3, range(training(3,:))+1);
for i = 1:length(PMF_3)
    PMF_3(1,i) = min_3+i-1;
    if(~isempty(find(tab_3_H0(:,1) == PMF_3(1,i))))
        PMF_3(3,i) = tab_3_H0(find(tab_3_H0(:,1) == PMF_3(1,i)), 3)/100;
    end
    if(~isempty(find(tab_3_H1(:,1) == PMF_3(1,i))))
        PMF_3(2,i) = tab_3_H1(find(tab_3_H1(:,1) == PMF_3(1,i)), 3)/100;
    end
end
for i = 1:length(PMF_3)
    if(PMF_3(3,i) == 0 && PMF_3(2,i) == 0)
        PMF_3(:,i) = [NaN; NaN; NaN];
    end
end
PMF_3 = rmmissing(PMF_3,2);


PMF_4 = zeros(3, range(training(4,:))+1);
for i = 1:length(PMF_4)
    PMF_4(1,i) = min_4+i-1;
    if(~isempty(find(tab_4_H0(:,1) == PMF_4(1,i))))
        PMF_4(3,i) = tab_4_H0(find(tab_4_H0(:,1) == PMF_4(1,i)), 3)/100;
    end
    if(~isempty(find(tab_4_H1(:,1) == PMF_4(1,i))))
        PMF_4(2,i) = tab_4_H1(find(tab_4_H1(:,1) == PMF_4(1,i)), 3)/100;
    end
end
for i = 1:length(PMF_4)
    if(PMF_4(3,i) == 0 && PMF_4(2,i) == 0)
        PMF_4(:,i) = [NaN; NaN; NaN];
    end
end
PMF_4 = rmmissing(PMF_4,2);

PMF_5 = zeros(3, range(training(5,:))+1);
for i = 1:length(PMF_5)
    PMF_5(1,i) = min_5+i-1;
    if(~isempty(find(tab_5_H0(:,1) == PMF_5(1,i))))
        PMF_5(3,i) = tab_5_H0(find(tab_5_H0(:,1) == PMF_5(1,i)), 3)/100;
    end
    if(~isempty(find(tab_5_H1(:,1) == PMF_5(1,i))))
        PMF_5(2,i) = tab_5_H1(find(tab_5_H1(:,1) == PMF_5(1,i)), 3)/100;
    end
end
for i = 1:length(PMF_5)
    if(PMF_5(3,i) == 0 && PMF_5(2,i) == 0)
        PMF_5(:,i) = [NaN; NaN; NaN];
    end
end
PMF_5 = rmmissing(PMF_5,2);

PMF_6 = zeros(3, range(training(6,:))+1);
for i = 1:length(PMF_6)
    PMF_6(1,i) = min_6+i-1;
    if(~isempty(find(tab_6_H0(:,1) == PMF_6(1,i))))
        PMF_6(3,i) = tab_6_H0(find(tab_6_H0(:,1) == PMF_6(1,i)), 3)/100;
    end
    if(~isempty(find(tab_6_H1(:,1) == PMF_6(1,i))))
        PMF_6(2,i) = tab_6_H1(find(tab_6_H1(:,1) == PMF_6(1,i)), 3)/100;
    end
end
for i = 1:length(PMF_6)
    if(PMF_6(3,i) == 0 && PMF_6(2,i) == 0)
        PMF_6(:,i) = [NaN; NaN; NaN];
    end
end
PMF_6 = rmmissing(PMF_6,2);

PMF_7 = zeros(3, range(training(7,:))+1);
for i = 1:length(PMF_7)
    PMF_7(1,i) = min_7+i-1;
    if(~isempty(find(tab_7_H0(:,1) == PMF_7(1,i))))
        PMF_7(3,i) = tab_7_H0(find(tab_7_H0(:,1) == PMF_7(1,i)), 3)/100;
    end
    if(~isempty(find(tab_7_H1(:,1) == PMF_7(1,i))))
        PMF_7(2,i) = tab_7_H1(find(tab_7_H1(:,1) == PMF_7(1,i)), 3)/100;
    end
end
for i = 1:length(PMF_7)
    if(PMF_7(3,i) == 0 && PMF_7(2,i) == 0)
        PMF_7(:,i) = [NaN; NaN; NaN];
    end
end
PMF_7 = rmmissing(PMF_7,2);
%% Uncomment this whole section below to graph values
% subplot(7,1,1);
% PMF_t = transpose(PMF_1);
% bar_1 = bar([1:1:length(PMF_1)], [PMF_t(:,3) PMF_t(:,2)], 'grouped');
% xlim([0 length(PMF_1)+1]);
% xticks([1:1:length(PMF_1)]);
% xticklabels(PMF_1(1,:));
% title('Mean Area Under Heart Beat');
% legend('H0', 'H1');
% 
% subplot(7,1,2);
% PMF_t = transpose(PMF_2);
% bar_2 = bar([1:1:length(PMF_2)], [PMF_t(:,3) PMF_t(:,2)], 'grouped');
% xlim([0 length(PMF_2)+1]);
% xticks([1:1:length(PMF_2)]);
% xticklabels(PMF_2(1,:));
% title('Mean R-to-R Peak Interval');
% 
% subplot(7,1,3);
% PMF_t = transpose(PMF_3);
% bar_3 = bar([1:1:length(PMF_3)], [PMF_t(:,3) PMF_t(:,2)], 'grouped');
% xlim([0 length(PMF_3)+1]);
% xticks([1:1:length(PMF_3)]);
% xticklabels(PMF_3(1,:));
% title('Heart Rate');
% 
% subplot(7,1,4);
% PMF_t = transpose(PMF_4);
% bar_4 = bar([1:1:length(PMF_4)], [PMF_t(:,3) PMF_t(:,2)], 'grouped');
% xlim([0 length(PMF_4)+1]);
% xticks([1:1:length(PMF_4)]);
% xticklabels(PMF_4(1,:));
% title('Peak-to-Peak Interval for Heart Pressure');
% 
% subplot(7,1,5);
% PMF_t = transpose(PMF_5);
% bar_5 = bar([1:1:length(PMF_5)], [PMF_t(:,3) PMF_t(:,2)], 'grouped');
% xlim([0 length(PMF_5)+1]);
% xticks([1:1:length(PMF_5)]);
% xticklabels(PMF_5(1,:));
% title('Systolic Blood Pressure');
% 
% subplot(7,1,6);
% PMF_t = transpose(PMF_6);
% bar_6 = bar([1:1:length(PMF_6)], [PMF_t(:,3) PMF_t(:,2)], 'grouped');
% xlim([0 length(PMF_6)+1]);
% xticks([1:1:length(PMF_6)]);
% xticklabels(PMF_6(1,:));
% title('Diastolic Blood Pressure');
% 
% subplot(7,1,7);
% PMF_t = transpose(PMF_7);
% bar_7 = bar([1:1:length(PMF_7)], [PMF_t(:,3) PMF_t(:,2)], 'grouped');
% xlim([0 length(PMF_7)+1]);
% xticks([1:1:length(PMF_7)]);
% xticklabels(PMF_7(1,:));
% title('Pulse Pressure');
%% Task 1 continued
HT_entry = zeros(length(PMF_1), 5);
HT_entry(:,1) = transpose(PMF_1(1,:));
HT_entry(:,2) = transpose(PMF_1(2,:));
HT_entry(:,3) = transpose(PMF_1(3,:));
decision = zeros(1,length(PMF_1));
MAP_1 = PMF_1;
MAP_1(2,:) = MAP_1(2,:)*P_H1;
MAP_1(3,:) = MAP_1(3,:)*P_H0;
for i=1:length(PMF_1)
    if(PMF_1(2,i) >= PMF_1(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
ML_1 = [PMF_1; decision];
HT_entry(:,4) = transpose(decision);
for i=1:length(MAP_1)
    if(MAP_1(2,i) >= MAP_1(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
MAP_1 = [MAP_1; decision];
HT_entry(:,5) = transpose(decision);
HT_table_array{patient, 1} = HT_entry;

HT_entry = zeros(length(PMF_2), 5);
HT_entry(:,1) = transpose(PMF_2(1,:));
HT_entry(:,2) = transpose(PMF_2(2,:));
HT_entry(:,3) = transpose(PMF_2(3,:));
decision = zeros(1,length(PMF_2));
MAP_2 = PMF_2;
MAP_2(2,:) = MAP_2(2,:)*P_H1;
MAP_2(3,:) = MAP_2(3,:)*P_H0;
for i=1:length(PMF_2)
    if(PMF_2(2,i) >= PMF_2(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
ML_2 = [PMF_2; decision];
HT_entry(:,4) = transpose(decision);
for i=1:length(MAP_2)
    if(MAP_2(2,i) >= MAP_2(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
MAP_2 = [MAP_2; decision];HT_entry(:,5) = transpose(decision);
HT_entry(:,5) = transpose(decision);
HT_table_array{patient, 2} = HT_entry;

HT_entry = zeros(length(PMF_3), 5);
HT_entry(:,1) = transpose(PMF_3(1,:));
HT_entry(:,2) = transpose(PMF_3(2,:));
HT_entry(:,3) = transpose(PMF_3(3,:));
decision = zeros(1,length(PMF_3));
MAP_3 = PMF_3;
MAP_3(2,:) = MAP_3(2,:)*P_H1;
MAP_3(3,:) = MAP_3(3,:)*P_H0;
for i=1:length(PMF_3)
    if(PMF_3(2,i) >= PMF_3(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
ML_3 = [PMF_3; decision];
HT_entry(:,4) = transpose(decision);
for i=1:length(MAP_3)
    if(MAP_3(2,i) >= MAP_3(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
MAP_3 = [MAP_3; decision];HT_entry(:,5) = transpose(decision);
HT_entry(:,5) = transpose(decision);
HT_table_array{patient, 3} = HT_entry;

HT_entry = zeros(length(PMF_4), 5);
HT_entry(:,1) = transpose(PMF_4(1,:));
HT_entry(:,2) = transpose(PMF_4(2,:));
HT_entry(:,3) = transpose(PMF_4(3,:));
decision = zeros(1,length(PMF_4));
MAP_4 = PMF_4;
MAP_4(2,:) = MAP_4(2,:)*P_H1;
MAP_4(3,:) = MAP_4(3,:)*P_H0;
for i=1:length(PMF_4)
    if(PMF_4(2,i) >= PMF_4(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
ML_4 = [PMF_4; decision];
HT_entry(:,4) = transpose(decision);
for i=1:length(MAP_4)
    if(MAP_4(2,i) >= MAP_4(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
MAP_4 = [MAP_4; decision];HT_entry(:,5) = transpose(decision);
HT_entry(:,5) = transpose(decision);
HT_table_array{patient, 4} = HT_entry;

HT_entry = zeros(length(PMF_5), 5);
HT_entry(:,1) = transpose(PMF_5(1,:));
HT_entry(:,2) = transpose(PMF_5(2,:));
HT_entry(:,3) = transpose(PMF_5(3,:));
decision = zeros(1,length(PMF_5));
MAP_5 = PMF_5;
MAP_5(2,:) = MAP_5(2,:)*P_H1;
MAP_5(3,:) = MAP_5(3,:)*P_H0;
for i=1:length(PMF_5)
    if(PMF_5(2,i) >= PMF_5(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
ML_5 = [PMF_5; decision];
HT_entry(:,4) = transpose(decision);
for i=1:length(MAP_5)
    if(MAP_5(2,i) >= MAP_5(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
MAP_5 = [MAP_5; decision];HT_entry(:,5) = transpose(decision);
HT_entry(:,5) = transpose(decision);
HT_table_array{patient, 5} = HT_entry;

HT_entry = zeros(length(PMF_6), 5);
HT_entry(:,1) = transpose(PMF_6(1,:));
HT_entry(:,2) = transpose(PMF_6(2,:));
HT_entry(:,3) = transpose(PMF_6(3,:));
decision = zeros(1,length(PMF_6));
MAP_6 = PMF_6;
MAP_6(2,:) = MAP_6(2,:)*P_H1;
MAP_6(3,:) = MAP_6(3,:)*P_H0;
for i=1:length(PMF_6)
    if(PMF_6(2,i) >= PMF_6(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
ML_6 = [PMF_6; decision];
HT_entry(:,4) = transpose(decision);
for i=1:length(MAP_6)
    if(MAP_6(2,i) >= MAP_6(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
MAP_6 = [MAP_6; decision];HT_entry(:,5) = transpose(decision);
HT_entry(:,5) = transpose(decision);
HT_table_array{patient, 6} = HT_entry;

HT_entry = zeros(length(PMF_7), 5);
HT_entry(:,1) = transpose(PMF_7(1,:));
HT_entry(:,2) = transpose(PMF_7(2,:));
HT_entry(:,3) = transpose(PMF_7(3,:));
decision = zeros(1,length(PMF_7));
MAP_7 = PMF_7;
MAP_7(2,:) = MAP_7(2,:)*P_H1;
MAP_7(3,:) = MAP_7(3,:)*P_H0;
for i=1:length(PMF_7)
    if(PMF_7(2,i) >= PMF_7(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
ML_7 = [PMF_7; decision];
HT_entry(:,4) = transpose(decision);
for i=1:length(MAP_7)
    if(MAP_7(2,i) >= MAP_7(3,i))
        decision(1,i) = 1;
    else
        decision(1,i) = 0;
    end
end
MAP_7 = [MAP_7; decision];HT_entry(:,5) = transpose(decision);
HT_entry(:,5) = transpose(decision);
HT_table_array{patient, 7} = HT_entry;
end