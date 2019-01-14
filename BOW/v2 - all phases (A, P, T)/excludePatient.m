function ySet =  excludePatient(xSet, lesion, patientIdx)
% exclude one Patient

%% test
noOfPatients = [17, 33, 23, 11, 11]
lesions = {'FNH', 'HCC', 'HMG', 'METAHIPER', 'METAHIPO'}

if patientIdx > noOfPatients(lesion)
    error('patientIndexExceeded');
end

%% find
stringToFind = strcat(lesions(lesion),num2str(patientIdx))
logical_cells = find(contains(xSet(1,lesion).ImageLocation,stringToFind));
if length(logical_cells) < 5
    disp('Less than 5 deleted!')
    disp (stringToFind)
end

%% delete
xSet(1, lesion).ImageLocation(:, logical_cells(1):logical_cells(end)) = [];

%% return
ySet = xSet;
