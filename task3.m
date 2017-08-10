        function [Joint_HT_table_out, Joint_HT_H0_out, Joint_HT_H1_out, ML_Alarms_out, MAP_Alarms_out, JT_Array_out, testing_alarms_ML, testing_alarms_MAP, testing_labels_ROC] = task3(patient_num, feature_num_1, feature_num_2, HT_table_array, prior_table, testing_set , label_testing)
%task3 - A function that performs the following functions:
%   a) Generates the joint likelihood matrices
%   b) Calculates ML and MAP decision vectors
%   c) Saves the results as a Joint_HT_table
%   d) Generate joint PMF meshes
%
%Inputs:
%   patient_num - the number of the patient you want to generate for (1-9)
%   feature_num_X - the numbers of the features you want to compare (1-7)
%   HT_table_array - the HT_table_array generated
%   prior_table - the prior_table generated earlier
%   testing - the data to be tested on
%   label_testing - the labels of the testing data to be tested on
%Outputs:
%   Joint_HT_table_out - the joint PMF with decision vectors formatted as
%   specified
%   Joint_HT_HX_out - a 2D matrix of the PMF values given a hypothesis. The
%   output is formatted where column indices correspond to feature_2
%   indicies, and row indices correspond to feature_1 indices (just to make
%   mesh work)
%   ML/MAP_Alarms_out - a 2D matrix of the decision vectors. The output is
%   formatted where column indices correspond to feature_2 indices, and row
%   indices correspond to feature_1 indices

feature = cell(7);
feature{1} = ' Mean Area Under the Heart Beat';
feature{2} = ' Mean R-to-R Peak Interval';
feature{3} = ' Heart Rate';
feature{4} = ' Peak-to-Peak Interval for Blood Pressure';
feature{5} = ' Systolic Blood Pressure';
feature{6} = ' Diastolic Blood Pressure';
feature{7} = ' Pulse Pressure';
Joint_HT_table = zeros((length(HT_table_array{patient_num, feature_num_1}(:,1))*length(HT_table_array{patient_num, feature_num_2}(:,1))), 6);
Joint_HT_H0 = zeros(length(HT_table_array{patient_num, feature_num_2}(:,1)), length(HT_table_array{patient_num, feature_num_1}(:,1)));
Joint_HT_H1 = zeros(length(HT_table_array{patient_num, feature_num_2}(:,1)), length(HT_table_array{patient_num, feature_num_1}(:,1)));
ML_Alarms = zeros(length(HT_table_array{patient_num, feature_num_2}(:,1)), length(HT_table_array{patient_num, feature_num_1}(:,1)));
MAP_Alarms = zeros(length(HT_table_array{patient_num, feature_num_2}(:,1)), length(HT_table_array{patient_num, feature_num_1}(:,1)));

row = 1;
for i=1:length(HT_table_array{patient_num, feature_num_1}(:,1))
    for j=1:length(HT_table_array{patient_num, feature_num_2}(:,1))
        Joint_HT_table(row,1) = HT_table_array{patient_num, feature_num_1}(i, 1);
        Joint_HT_table(row,2) = HT_table_array{patient_num, feature_num_2}(j, 1);
        Joint_HT_table(row,3) = HT_table_array{patient_num, feature_num_1}(i, 2) * HT_table_array{patient_num, feature_num_2}(j, 2);
        Joint_HT_H0(j,i) = HT_table_array{patient_num, feature_num_1}(i, 2) * HT_table_array{patient_num, feature_num_2}(j, 2);
        Joint_HT_table(row,4) = HT_table_array{patient_num, feature_num_1}(i, 3) * HT_table_array{patient_num, feature_num_2}(j, 3);
        Joint_HT_H1(j,i) = HT_table_array{patient_num, feature_num_1}(i, 3) * HT_table_array{patient_num, feature_num_2}(j, 3);
        if(Joint_HT_table(row, 3) >= Joint_HT_table(row, 4))
            Joint_HT_table(row, 5) = 1;
            ML_Alarms(j, i) = 1;
        else
            Joint_HT_table(row, 5) = 0;
            ML_Alarms(j, i) = 0;
        end
        if(Joint_HT_table(row, 3)*prior_table(2, patient_num) >= Joint_HT_table(row, 4)*prior_table(1, patient_num))
            Joint_HT_table(row, 6) = 1;
            MAP_Alarms(j, i) = 1;
        else
            Joint_HT_table(row, 6) = 0;
            MAP_Alarms(j, i) = 0;
        end
        row = row+1;
    end
end

hold on;
subplot(2,1,1);
mesh(HT_table_array{patient_num, feature_num_1}(:,1), HT_table_array{patient_num, feature_num_2}(:,1), Joint_HT_H0);
title(strcat(feature{feature_num_1}, ' and', feature{feature_num_2}, ' Patient\_', num2str(patient_num), ' (H0)'));
subplot(2,1,2);
mesh(HT_table_array{patient_num, feature_num_1}(:,1), HT_table_array{patient_num, feature_num_2}(:,1), Joint_HT_H1);
title(strcat(feature{feature_num_1}, ' and', feature{feature_num_2}, ' Patient\_', num2str(patient_num), ' (H1)'));

Joint_HT_table_out = Joint_HT_table;
Joint_HT_H0_out = Joint_HT_H0;
Joint_HT_H1_out = Joint_HT_H1;
ML_Alarms_out = ML_Alarms;
MAP_Alarms_out = MAP_Alarms;

% figure
% mesh(HT_table_array{patient_num, feature_num_1}(:,1), HT_table_array{patient_num, feature_num_2}(:,1), ML_Alarms);
count_FA_ML = 0;
count_MD_ML = 0;
count_FA_MAP = 0;
count_MD_MAP = 0;


testing_alarms_ML = zeros(1, length(label_testing));
testing_alarms_MAP = zeros(1, length(label_testing));
testing_labels_ROC = zeros(1, length(label_testing));


for i=1:length(label_testing)

    column = find(HT_table_array{patient_num, feature_num_1}(:,1) == testing_set(feature_num_1, i), 1, 'first');
    row = find(HT_table_array{patient_num, feature_num_2}(:,1) == testing_set(feature_num_2, i), 1, 'first');
    ML_alarm = ML_Alarms(row, column);
    MAP_alarm = MAP_Alarms(row, column);
    
    %generates a alarm array for Joint ML Rule
    testing_alarms_ML(1, i) = ML_alarm;
    testing_alarms_MAP(1, i) = MAP_alarm;
    
    
    testing_labels_ROC(1, i) = Joint_HT_H1(row, column);
   
   
    %Count of False Alarm for ML
    if(ML_alarm == 1 && label_testing(i) == 0)
        count_FA_ML = count_FA_ML + 1;
    end
    
    %Count of Miss Detection for ML
    if(ML_alarm == 0 && label_testing(i) == 1)
        count_MD_ML = count_MD_ML + 1;
    end
    
    %Count of Fa for MAP
    if(MAP_alarm == 1 && label_testing(i) == 0)
        count_FA_MAP = count_FA_MAP + 1;
    end
    
   %Count for MD for MAP 
    if(MAP_alarm == 0 && label_testing(i) == 1)
        count_MD_MAP = count_MD_MAP + 1;
    end
    
end

label_testing_stat = tabulate(label_testing);
JT_Array = zeros(2,3);

JT_Array(1,1) = count_FA_ML/label_testing_stat(1,2);
JT_Array(1,2) = count_MD_ML/label_testing_stat(2,2);
JT_Array(1,3) = (count_FA_ML + count_MD_ML)/(label_testing_stat(1,2) + label_testing_stat(2,2));

JT_Array(2,1) = count_FA_MAP/label_testing_stat(1,2);
JT_Array(2,2) = count_MD_MAP/label_testing_stat(2,2);
JT_Array(2,3) = (count_FA_MAP + count_MD_MAP)/(label_testing_stat(1,2) + label_testing_stat(2,2));

JT_Array_out = JT_Array;