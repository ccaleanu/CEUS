function [ySet, testImgs] =  excludePatient(xSet, lesion, patientIdx)
% exclude one Patient

s = inputname(1);
disp(['Parsing ''' s '''...'])

%% test
noOfPatients = [17, 33, 23, 11, 11];
lesions = {'FNH', 'HCC', 'HMG', 'METAHIPER', 'METAHIPO'};

if patientIdx > noOfPatients(lesion)
    error('patientIndexExceeded');
end

%% find
stringToFind = strcat(lesions(lesion),num2str(patientIdx));
logical_cells = find(contains(xSet(1,lesion).ImageLocation,stringToFind));
if strcmp(s, 'imgSetsA') && (length(logical_cells) < 5)
    disp('Less than 5 deleted!')
    disp (stringToFind)
end
if strcmp(s, 'imgSetsP') && (length(logical_cells) < 3)
    disp('Less than 3 deleted!')
    disp (stringToFind)
end
if strcmp(s, 'imgSetsT') && (length(logical_cells) < 2)
    disp('Less than 2 deleted!')
    disp (stringToFind)
end

%% get
testImgs = xSet(1, lesion).ImageLocation(:, logical_cells(1):logical_cells(end));

%% delete
xSet(1, lesion).ImageLocation(:, logical_cells(1):logical_cells(end)) = [];

%% return

disp('done')
ySet = xSet;
