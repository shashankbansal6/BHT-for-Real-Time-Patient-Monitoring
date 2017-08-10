%% Setup the HT_table_array before stepping into the loop
clear;
clc;

HT_table_array = cell(9,7);

%these will contain the golden alarms for each patients as per the decision
%rule
golden_alarm_table_ML = cell(9,1);
golden_alarm_table_MAP = cell(9,1);
training_data_array = cell(2,9);
testing_data_array = cell(2,9);
tab_table_array = cell(1,9);
prior_table = [];
range_table_array = cell(9,7);
training_struct = struct([]);
testing_struct = struct([]);

Error_table_array = cell(9,7);
%% Task 0 - Loading data
%This loads all the patient data into a structure of patients
patient_data_struct(1) = load('1_a41178.mat'); % Patient 1
patient_data_struct(2) = load('2_a42126.mat'); % Patient 2
patient_data_struct(3) = load('3_a40076.mat'); % Patient 3
patient_data_struct(4) = load('4_a40050.mat'); % Patient 4
patient_data_struct(5) = load('5_a41287.mat'); % Patient 5
patient_data_struct(6) = load('6_a41846.mat'); % Patient 6
patient_data_struct(7) = load('7_a41846.mat'); % Patient 7
patient_data_struct(8) = load('8_a42008.mat'); % Patient 8
patient_data_struct(9) = load('9_a41846.mat'); % Patient 9

for patient=1:9
    all_data = floor(patient_data_struct(patient).all_data);
    all_labels = patient_data_struct(patient).all_labels;
    for i=1:7
        range_table_array{patient, i} = [min(all_data(i,:)); max(all_data(i,:))];
    end

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

    training_data_array{1, patient} = training;
    training_data_array{2, patient} = label_training;
    testing_data_array{1, patient} = testing;
    testing_data_array{2, patient} = label_testing;
    training_struct(patient).data = training;
    training_struct(patient).labels = label_training;
    testing_struct(patient).data = testing;
    testing_struct(patient).labels = label_testing;
    %% Task 1.1 - Enter the Likelihood Matrix
    % Start off by calculating the prior probabilities, and setting up the H0
    % and H1 training data
    tabulation = tabulate(label_training);
    training_H0 = [];
    training_H1 = [];
    P_H0 = tabulation(1,3)/100;
    P_H1 = tabulation(2,3)/100;
    prior_table = [prior_table [P_H0; P_H1]];

    % The idea is that we'll separate golden alarm data from non-golden alarm
    % data. This is version-safe. Basically, it builds the matrices column by
    % column.
    for i = 1:length(label_training)
        if(label_training(1,i) == 0)
            training_H0 = [training_H0 training(:,i)];
        else
            training_H1 = [training_H1 training(:,i)];
        end
    end

    % We figure out minimum, maximum, and range of values for later use
    mmr_table = zeros(3,7);
    for i=1:7
        mmr_table(:,i) = [min(training(i,:)); max(training(i,:)); range(training(i,:))];
    end

    % We tabulate given H0 and H1 for our likelihood matrix. This tabulation
    % table stores every tabulation where row 1 are H0 tabulations, row 2 are
    % H1 tabulations, and each column goes with each characteristic. It'll be
    % switched over to this for the sake of neatness. Maybe. For now, the
    % individual tabulations will be kept.
    tab_table = cell(2,7);
    for i=1:7
        tab_table{1,i} = tabulate(training_H0(i,:));
        tab_table{2,i} = tabulate(training_H1(i,:));
    end
    tab_table_array{patient} = tab_table;
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

    % Each PMF is generated. Going line by line, first, it creates a matrix
    % with 3 rows and as many columns as data values for the data. Second, it
    % does two searches. Without loss of generality, for each value in the data
    % set, it finds the right probability associated with the value and fills
    % in the appropriate row based on whether the probability is associated
    % with H1 or H0.
    % In other words, Row 1 is filled with all the values the characteristic
    % could take. Row 2 is filled with the probabilities given H1. Row 3 is
    % filled with the probabilities given H0.
    PMF_1 = zeros(3, range(range_table_array{patient,1})+1);
    PMF_1(1,:) = range_table_array{patient,1}(1):1:range_table_array{patient,1}(2);
    for i = 1:length(PMF_1)
        if(~isempty(find(tab_1_H0(:,1) == PMF_1(1,i))))
            PMF_1(3,i) = tab_1_H0(find(tab_1_H0(:,1) == PMF_1(1,i)), 3)/100;
        end
        if(~isempty(find(tab_1_H1(:,1) == PMF_1(1,i))))
            PMF_1(2,i) = tab_1_H1(find(tab_1_H1(:,1) == PMF_1(1,i)), 3)/100;
        end
    end

    PMF_2 = zeros(3, range(range_table_array{patient,2})+1);
    PMF_2(1,:) = range_table_array{patient,2}(1):1:range_table_array{patient,2}(2);
    for i = 1:length(PMF_2)
        if(~isempty(find(tab_2_H0(:,1) == PMF_2(1,i))))
            PMF_2(3,i) = tab_2_H0(find(tab_2_H0(:,1) == PMF_2(1,i)), 3)/100;
        end
        if(~isempty(find(tab_2_H1(:,1) == PMF_2(1,i))))
            PMF_2(2,i) = tab_2_H1(find(tab_2_H1(:,1) == PMF_2(1,i)), 3)/100;
        end
    end

    PMF_3 = zeros(3, range(range_table_array{patient,3})+1);
    PMF_3(1,:) = range_table_array{patient,3}(1):1:range_table_array{patient,3}(2);
    for i = 1:length(PMF_3)
        if(~isempty(find(tab_3_H0(:,1) == PMF_3(1,i))))
            PMF_3(3,i) = tab_3_H0(find(tab_3_H0(:,1) == PMF_3(1,i)), 3)/100;
        end
        if(~isempty(find(tab_3_H1(:,1) == PMF_3(1,i))))
            PMF_3(2,i) = tab_3_H1(find(tab_3_H1(:,1) == PMF_3(1,i)), 3)/100;
        end
    end

    PMF_4 = zeros(3, range(range_table_array{patient,4})+1);
    PMF_4(1,:) = range_table_array{patient,4}(1):1:range_table_array{patient,4}(2);
    for i = 1:length(PMF_4)
        if(~isempty(find(tab_4_H0(:,1) == PMF_4(1,i))))
            PMF_4(3,i) = tab_4_H0(find(tab_4_H0(:,1) == PMF_4(1,i)), 3)/100;
        end
        if(~isempty(find(tab_4_H1(:,1) == PMF_4(1,i))))
            PMF_4(2,i) = tab_4_H1(find(tab_4_H1(:,1) == PMF_4(1,i)), 3)/100;
        end
    end

    PMF_5 = zeros(3, range(range_table_array{patient,5})+1);
    PMF_5(1,:) = range_table_array{patient,5}(1):1:range_table_array{patient,5}(2);
    for i = 1:length(PMF_5)
        if(~isempty(find(tab_5_H0(:,1) == PMF_5(1,i))))
            PMF_5(3,i) = tab_5_H0(find(tab_5_H0(:,1) == PMF_5(1,i)), 3)/100;
        end
        if(~isempty(find(tab_5_H1(:,1) == PMF_5(1,i))))
            PMF_5(2,i) = tab_5_H1(find(tab_5_H1(:,1) == PMF_5(1,i)), 3)/100;
        end
    end

    PMF_6 = zeros(3, range(range_table_array{patient,6})+1);
    PMF_6(1,:) = range_table_array{patient,6}(1):1:range_table_array{patient,6}(2);
    for i = 1:length(PMF_6)
        if(~isempty(find(tab_6_H0(:,1) == PMF_6(1,i))))
            PMF_6(3,i) = tab_6_H0(find(tab_6_H0(:,1) == PMF_6(1,i)), 3)/100;
        end
        if(~isempty(find(tab_6_H1(:,1) == PMF_6(1,i))))
            PMF_6(2,i) = tab_6_H1(find(tab_6_H1(:,1) == PMF_6(1,i)), 3)/100;
        end
    end

    PMF_7 = zeros(3, range(range_table_array{patient,7})+1);
    PMF_7(1,:) = range_table_array{patient,7}(1):1:range_table_array{patient,7}(2);
    for i = 1:length(PMF_7)
        if(~isempty(find(tab_7_H0(:,1) == PMF_7(1,i))))
            PMF_7(3,i) = tab_7_H0(find(tab_7_H0(:,1) == PMF_7(1,i)), 3)/100;
        end
        if(~isempty(find(tab_7_H1(:,1) == PMF_7(1,i))))
            PMF_7(2,i) = tab_7_H1(find(tab_7_H1(:,1) == PMF_7(1,i)), 3)/100;
        end
    end

    % This is the part where each data set is graphed. Because of style, we
    % the plots will be lines, but these are still PMFs. The style could be
    % changed later on if desired (i.e. we ask a TA)
    % set(gcf, 'Visible', 'off')
    % set(0, 'DefaultFigureVisible', 'off');
    % figure;
    % subplot(7,1,1);
    % plot(PMF_1(1,:), PMF_1(3,:), 'b', PMF_1(1,:), PMF_1(2,:), 'r');
    % legend('H0', 'H1');
    % title(strcat('Mean Area Under Heart Beat (Patient\_', num2str(patient), ')'));
    % grid on;
    % % fname = strcat('Mean Area Under Heart Beat_', num2str(patient));
    % % print(fname, '-dpng');
    % 
    % subplot(7,1,2);
    % plot(PMF_2(1,:), PMF_2(3,:), 'b', PMF_2(1,:), PMF_2(2,:), 'r');
    % legend('H0', 'H1');
    % title(strcat('Mean R-to-R Peak Interval (Patient\_', num2str(patient), ')'));
    % grid on;
    % % fname = strcat('Mean R-to-R Peak Interval_', num2str(patient));
    % % print(fname, '-dpng');
    % 
    % subplot(7,1,3);
    % plot(PMF_3(1,:), PMF_3(3,:), 'b', PMF_3(1,:), PMF_3(2,:), 'r');
    % legend('H0', 'H1');
    % title(strcat('Heart Rate (Patient\_', num2str(patient), ')'));
    % grid on;
    % % fname = strcat('Heart Rate_', num2str(patient));
    % % print(fname, '-dpng');
    % 
    % subplot(7,1,4);
    % plot(PMF_4(1,:), PMF_4(3,:), 'b', PMF_4(1,:), PMF_4(2,:), 'r');
    % legend('H0', 'H1');
    % title(strcat('Peak-to-Peak Interval for Heart Pressure (Patient\_', num2str(patient), ')'));
    % grid on;
    % % fname = strcat('Peak-to-Peak Interval for Heart Pressure_', num2str(patient));
    % % print(fname, '-dpng');
    % 
    % subplot(7,1,5);
    % plot(PMF_5(1,:), PMF_5(3,:), 'b', PMF_5(1,:), PMF_5(2,:), 'r');
    % legend('H0', 'H1');
    % title(strcat('Systolic Blood Pressure (Patient\_', num2str(patient), ')'));
    % grid on;
    % % fname = strcat('Systolic Blood Pressure_', num2str(patient));
    % % print(fname, '-dpng');
    % 
    % subplot(7,1,6);
    % plot(PMF_6(1,:), PMF_6(3,:), 'b', PMF_6(1,:), PMF_6(2,:), 'r');
    % legend('H0', 'H1');
    % title(strcat('Diastolic Blood Pressure (Patient\_', num2str(patient), ')'));
    % grid on;
    % % fname = strcat('Diastolic Blood Pressure_', num2str(patient));
    % % print(fname, '-dpng');
    % 
    % subplot(7,1,7);
    % plot(PMF_7(1,:), PMF_7(3,:), 'b', PMF_7(1,:), PMF_7(2,:), 'r');
    % legend('H0', 'H1');
    % title(strcat('Pulse Pressure (Patient\_', num2str(patient), ')'));
    % grid on;
    % % fname = strcat('Pulse Pressure_', num2str(patient));
    % % print(fname, '-dpng');

    % We now begin to create entries for the HT_table_array. The idea is, we'll
    % create an HT_entry with as many rows as data values in the tabulation,
    % and 5 columns, which, in order go: Value X, P(X|H1), P(X|H0), ML
    % decision, and MAP decision
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
    %% Task 1.2
    %1.2 a and b

    %these stats are different for each patient but same for each feature 
    label_testing_stats = tabulate(label_testing);
    C_H0_T = label_testing_stats(1,2); %count of physician indicates no abnormality
    C_H1_T = label_testing_stats(2,2); %count of physician indicates an abnormality

    %golden alarms for each feature
    golden_alarm_ML = zeros(7, length(testing));
    golden_alarm_MAP = zeros(7, length(testing));

    for i = 1:7
        false_alarm_counter_ML = 0;
        false_alarm_counter_MAP = 0;
        miss_det_counter_ML = 0;
        miss_det_counter_MAP = 0;
        error_counter_ML = 0;
        error_counter_MAP = 0;
        error_table_temp = zeros(2,3);
        %first creating a temporary array to extract each of the feature data
        %from HT_table _array
        %In HT_decision_temp, the 4th column is for ML and the 5th column is
        %for MAP
        HT_decision_temp = HT_table_array{patient, i};

        %this loops through the length of testing and the for each feature goes
        %through all the columns of testing and finds in HT_decision_temp each column
        %value and its coresponding H0 or H1 based on ML and MAP decision rule 
        %In HT_decision_temp, the 4th column is for ML and the 5th column is
        %for MAP
        for k=1:length(testing)
            if(~isempty(find(testing(i, k) == HT_decision_temp(:, 1))))
                index = find(testing(i, k) == HT_decision_temp(:, 1));
                golden_alarm_ML(i, k) = HT_decision_temp(index, 4); %ML decision rule golden alarm
                golden_alarm_MAP(i, k) = HT_decision_temp(index, 5);%MAP decision rule golden alarm
            end

               %this is the false_alarm counter for ML rule which increases
               %count when physician indicates no abnormality but ML declares
               %an alarm
            if(golden_alarm_ML(i,k) == 1 && label_testing(1, k) == 0)
                false_alarm_counter_ML = false_alarm_counter_ML + 1;
            end

                %this is the miss detection counter for ML rule which increases
               %count when physician indicates an abnormality but ML declares
               %no alarm
            if(golden_alarm_ML(i,k) == 0 && label_testing(1, k) == 1)
                miss_det_counter_ML = miss_det_counter_ML + 1;
            end  

            if((golden_alarm_ML(i,k) == 1 && label_testing(1, k) == 0)||(golden_alarm_ML(i,k) == 0 && label_testing(1, k) == 1))
                error_counter_ML = error_counter_ML + 1;   
            end


            %this is the false_alarm probability for MAP rule which increases
               %count when physician indicates no abnormality but ML declares
               %an alarm
            if(golden_alarm_MAP(i,k) == 1 && label_testing(1, k) == 0)
                false_alarm_counter_MAP = false_alarm_counter_MAP + 1;
            end

                %this is the miss detection probability for MAP rule which increases
               %count when physician indicates an abnormality but ML declares
               %no alarm
            if(golden_alarm_MAP(i,k) == 0 && label_testing(1, k) == 1)
                miss_det_counter_MAP = miss_det_counter_MAP + 1;
            end

            if((golden_alarm_MAP(i,k) == 1 && label_testing(1, k) == 0)||(golden_alarm_MAP(i,k) == 0 && label_testing(1, k) == 1))
                error_counter_MAP = error_counter_MAP + 1;   
            end
        end

        %this forms a temporary 2X3 array table for each feature and puts in
        %the prob as required in task 1.2
        error_table_temp(1,1) = false_alarm_counter_ML/C_H0_T; %Prob. of false alarms for ML rule
        error_table_temp(1,2) = miss_det_counter_ML/C_H1_T; %Prob. of missed detection for MAP rule
        error_table_temp(1,3) = error_counter_ML/(C_H0_T+C_H1_T); %prob. of error for ML rule

        error_table_temp(2,1) = false_alarm_counter_MAP/C_H0_T; %Prob. of false alarms for MAP rule
        error_table_temp(2,2) = miss_det_counter_MAP/C_H1_T; %Prob. of miss detection for MAP rule
        error_table_temp(2,3) = error_counter_MAP/(C_H0_T+C_H1_T);
        %(label_testing_stats(1,3)/100)*(error_table_temp(2,1)) + (label_testing_stats(2,3)/100)*(error_table_temp(2,2));

        %this adds each 2X3 temp_error array to the Error_table_array for each
        %feature of each patient 
        Error_table_array{patient, i} = error_table_temp;
    end

    golden_alarm_table_ML{patient,1} = golden_alarm_ML;
    golden_alarm_table_MAP{patient,1} = golden_alarm_MAP;      
end   
%% Task 1 Cleanup
clear all_data all_labels HT_entry P_H0 P_H1 training_H0 training_H1 testing_idx training_idx tabulation training decision label_training label_testing
clear golden_alarm_MAP golden_alarm_ML error_counter_MAP error_counter_ML error_table_temp miss_det_counter_MAP miss_det_counter_ML
clear -regexp ^MAP_ ^ML_ ^PMF_
clear -regexp 'tab_\d_H\d'
    
%% Task 2.1
%Our goal is to find the problematic data which means that 2 patients have
%similar data. 
corr_cell = cell(36, 7);
counter = 1;
%the first 2 for loops are to find all the patient possibilities which is
% 9 choose 2 = 36 possibilities
for i = 1:8
    for j = i+1:9
        %this for loop is to find the correlation betweeen the patients and
        %each of the 7 features and add it to a cell
        for k = 1:7
            %the if statement compares the length of the 2 patient datas
            %and then cuts the larger one to the length of the smaller one
            if(length(patient_data_struct(i).all_data) <= length(patient_data_struct(j).all_data))
                A = patient_data_struct(i).all_data; 
                B = patient_data_struct(j).all_data(:, 1:length(A));
                R = corrcoef(A(k,:), B(k,:));
            else 
                B = patient_data_struct(j).all_data;
                A = patient_data_struct(i).all_data(:, 1:length(B));
                R = corrcoef(A(k,:), B(k,:));
            end
            corr_cell{counter, k} = R(1,2);
        end
        counter = counter + 1;
    end
end
clear counter A B R

%this converts the cells to a table format for better visualization
corr_T = cell2table(corr_cell, 'RowNames', {'Patient 1 and 2', 'Patient 1 and 3','Patient 1 and 4','Patient 1 and 5','Patient 1 and 6','Patient 1 and 7','Patient 1 and 8','Patient 1 and 9','Patient 2 and 3','Patient 2 and 4','Patient 2 and 5','Patient 2 and 6','Patient 2 and 7','Patient 2 and 8','Patient 2 and 9','Patient 3 and 4','Patient 3 and 5','Patient 3 and 6','Patient 3 and 7','Patient 3 and 8','Patient 3 and 9','Patient 4 and 5','Patient 4 and 6','Patient 4 and 7','Patient 4 and 8','Patient 4 and 9','Patient 5 and 6','Patient 5 and 7','Patient 5 and 8','Patient 5 and 9','Patient 6 and 7','Patient 6 and 8','Patient 6 and 9','Patient 7 and 8','Patient 7 and 9','Patient 8 and 9',}, 'VariableNames', {'MeanArea', 'MeanR2R', 'HeartRate','Peak2PeakInterval','SystolicBloodPressure','DiastolicBloodPressure','PulsePressure'});


%% Task 2.2 
%Technique 1

%choosing patients 1,5 and 7 because they have the most number of data
%points that is the number of columns

%this is the error list for patient 5
ML_error_list = zeros(7,3);
MAP_error_list = zeros(7,3);

for i = 1:7
    Error_temp_1 = Error_table_array{1, i};
    Error_temp_5 = Error_table_array{5, i};
    Error_temp_7 = Error_table_array{7, i};
    
    ML_error_list(i,1) =  Error_temp_1(1,3);
    MAP_error_list(i,1) = Error_temp_1(2,3);
    
    ML_error_list(i,2) =  Error_temp_5(1,3);
    MAP_error_list(i,2) = Error_temp_5(2,3);
    
    ML_error_list(i,3) =  Error_temp_7(1,3);
    MAP_error_list(i,3) = Error_temp_7(2,3);
end

%Since ML error are comparatively higher than MAP_errors, I am going to
%choose the features based on MAP_error list for all the patients. 
 


%Technique 2
%trying to find the correlation between features and then compare
%first lets extract patient training set which is saved in training_set_i

all_data_1 = floor(patient_data_struct(1).all_data);
all_data_5  = floor(patient_data_struct(5).all_data);
all_data_7  = floor(patient_data_struct(7).all_data);
all_labels_1 = patient_data_struct(1).all_labels;
all_labels_5 = patient_data_struct(5).all_labels;
all_labels_7 = patient_data_struct(7).all_labels;

training_idx_1 = floor(length(all_data_1) * 2/3);
training_idx_5 = floor(length(all_data_5) * 2/3);
training_idx_7 = floor(length(all_data_7) * 2/3);

training_set_1 = all_data_1(:, 1:training_idx_1);
testing_set_1 = all_data_1(:, training_idx_1:length(all_data_1));
testing_labels_1 = all_labels_1(:, training_idx_1:length(all_labels_1));
training_set_5 = all_data_5(:, 1:training_idx_5);
testing_set_5 = all_data_5(:, training_idx_5:length(all_data_5));
testing_labels_5 = all_labels_5(:, training_idx_5:length(all_labels_5));
training_set_7 = all_data_7(:, 1:training_idx_7);
testing_set_7 = all_data_7(:, training_idx_7:length(all_data_7));
testing_labels_7 = all_labels_7(:, training_idx_7:length(all_labels_7));

%Now lets find a table for the corelation between all the features of
%patient 1, 5 and 7
%the total possibilities are 7 choose 2
feature_corr = cell(21, 3);
counter = 0;
for i = 1:6
    for j = i+1:7
        corr_temp_1 = corrcoef(training_set_1(i,:), training_set_1(j, :));
        corr_temp_5 = corrcoef(training_set_5(i,:), training_set_5(j, :));
        corr_temp_7 = corrcoef(training_set_7(i,:), training_set_7(j, :));
        counter = counter + 1;
        feature_corr{counter, 1} = corr_temp_1(1,2);
        feature_corr{counter, 2} = corr_temp_5(1,2);
        feature_corr{counter, 3} = corr_temp_7(1,2);
    end
end


%this is the table of correlation for all the features for patients 1,5 and
%7
%the negative correlation means the as the feature a increases, feature b
%decreases and the magnitude represents the value of correlation
feature_corr_table = cell2table(feature_corr, 'RowNames', {'Feature 1 and 2', 'Feature 1 and 3','Feature 1 and 4','Feature 1 and 5','Feature 1 and 6','Feature 1 and 7','Feature 2 and 3','Feature 2 and 4','Feature 2 and 5','Feature 2 and 6','Feature 2 and 7','Feature 3 and 4','Feature 3 and 5','Feature 3 and 6','Feature 3 and 7','Feature 4 and 5','Feature 4 and 6','Feature 4 and 7','Feature 5 and 6','Feature 5 and 7','Feature 6 and 7'},'VariableNames', {'Correlation_Patient1','Correlation_Patient5','Correlation_Patient7'});

abs_feature_corr = abs(cell2mat(feature_corr));

%for patient 1, feature 6 and 7 have the least value for correlation
%for patient 5, feature 1 and 6 have the least value of correlation
%for patient 7, feature 3 and 5 have the least value of correlation

%% Task 3.1 and Task 3.2

probability_table = cell(2,3);

[Joint_HT_table_out, Joint_HT_H0_out, Joint_HT_H1_out, ML_Alarms_out, MAP_Alarms_out, JT_Array_out, testing_alarms_ML, testing_alarms_MAP, testing_labels_ROC] = task3(1, 5, 7, HT_table_array, prior_table, testing_set_1, testing_labels_1);

probability_table{1,1} = JT_Array_out;

figure;
hold on;
subplot(3,1,1);
bar(testing_alarms_ML);
title('ML Alarms from Testing dataset for Feature pair');

subplot(3,1,2);
bar(testing_alarms_MAP);
title('MAP Alarms from testing dataset for Feature pair');

subplot(3,1,3);
bar(testing_labels_1);
title('Golden Alarms');


[Joint_HT_table_out, Joint_HT_H0_out, Joint_HT_H1_out, ML_Alarms_out, MAP_Alarms_out, JT_Array_out, testing_alarms_ML, testing_alarms_MAP, testing_labels_ROC] = task3(5, 2, 3, HT_table_array, prior_table, testing_set_5, testing_labels_5);

probability_table{1,2} = JT_Array_out;
 
figure;
hold on;
subplot(3,1,1);
bar(testing_alarms_ML);
title('ML Alarms from Testing dataset for Feature pair');

subplot(3,1,2);
bar(testing_alarms_MAP);
title('MAP Alarms from testing dataset for Feature pair');

subplot(3,1,3);
bar(testing_labels_5);
title('Golden Alarms');

[Joint_HT_table_out, Joint_HT_H0_out, Joint_HT_H1_out, ML_Alarms_out, MAP_Alarms_out,JT_Array_out, testing_alarms_ML, testing_alarms_MAP, testing_labels_ROC] = task3(7, 3, 5, HT_table_array, prior_table, testing_set_7, testing_labels_7);

probability_table{1,3} = JT_Array_out;
figure;
hold on;
subplot(3,1,1);
bar(testing_alarms_ML);
title('ML Alarms from Testing dataset for Feature pair');

subplot(3,1,2);
bar(testing_alarms_MAP);
title('MAP Alarms from testing dataset for Feature pair');

subplot(3,1,3);
bar(testing_labels_7);
title('Golden Alarms');


%% Task 3.3

array1 = probability_table{1,1};
array2 = probability_table{1,2};
array3 = probability_table{1,3};
avg_Error_ML_testing = (array1(1,3) + array2(1,3) + array3(1,3))/3;
avg_Error_MAP_testing = (array1(2,3) + array2(2,3) + array3(2,3))/3;

%clear Joint_HT_table_out Joint_HT_H0_out Joint_HT_H1_out ML_Alarms_out MAP_Alarms_out JT_Array_out testing_alarms_ML testing_alarms_MAP;

%%Task 3.4
[Joint_HT_table_out, Joint_HT_H0_out, Joint_HT_H1_out, ML_Alarms_out, MAP_Alarms_out, JT_Array_out, testing_alarms_ML, testing_alarms_MAP, testing_labels_ROC] = task3(1, 5, 7, HT_table_array, prior_table, testing_set_1, testing_labels_1);

figure
hold on;

[X, Y] = perfcurve(testing_labels_1,testing_labels_ROC, 1);
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');

% figure
% hold on
% [A, B] = perfcurve(testing_alarms_MAP, testing_labels_1, 1);
% plot(A, B);

clear Joint_HT_table_out Joint_HT_H0_out Joint_HT_H1_out ML_Alarms_out MAP_Alarms_out JT_Array_out testing_alarms_ML testing_alarms_MAP testing_labels_ROC;

[Joint_HT_table_out, Joint_HT_H0_out, Joint_HT_H1_out, ML_Alarms_out, MAP_Alarms_out, JT_Array_out, testing_alarms_ML, testing_alarms_MAP, testing_labels_ROC] = task3(5, 2, 3, HT_table_array, prior_table, testing_set_5, testing_labels_5);
figure
hold on;

[X, Y] = perfcurve(testing_labels_5, testing_labels_ROC, 1);
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');

% figure
% hold on
% [A, B] = perfcurve(testing_alarms_MAP, testing_labels_5, 1);
% plot(A, B);

clear Joint_HT_table_out Joint_HT_H0_out Joint_HT_H1_out ML_Alarms_out MAP_Alarms_out JT_Array_out testing_alarms_ML testing_alarms_MAP testing_labels_ROC;

[Joint_HT_table_out, Joint_HT_H0_out, Joint_HT_H1_out, ML_Alarms_out, MAP_Alarms_out,JT_Array_out, testing_alarms_ML, testing_alarms_MAP, testing_labels_ROC] = task3(7, 3, 5, HT_table_array, prior_table, testing_set_7, testing_labels_7);

figure
hold on;

[X, Y] = perfcurve(testing_labels_7, testing_labels_ROC, 1);
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');

% figure
% hold on
% [A, B] = perfcurve(testing_alarms_MAP, testing_labels_7, 1);
% plot(A, B);