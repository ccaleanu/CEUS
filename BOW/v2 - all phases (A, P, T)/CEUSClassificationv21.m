%% New
% on v21
% exclude one Patient
% on v2:
% loop evaluation

%% Init
clear all
close all
clc
startTime = tic;

%%  Create an array of image sets from multiple folders
% get the dataset from:
% https://uptro29158-my.sharepoint.com/:f:/g/personal/intelligentembeddedvision_upt_ro/EgEsDIYU6XRMvqpt0wTzE9oBB5RUC9xR77XnbwoFtAhv2w?e=6SHDJ9
imgFolderA = '\MY\My Databases\CEUS\UMF\Picture DBV3 - A P T\A';
imgFolderP = '\MY\My Databases\CEUS\UMF\Picture DBV3 - A P T\P';
imgFolderT = '\MY\My Databases\CEUS\UMF\Picture DBV3 - A P T\T';

imgSetsA = imageSet(imgFolderA,'recursive');
imgSetsP = imageSet(imgFolderP,'recursive');
imgSetsT = imageSet(imgFolderT,'recursive');

{imgSetsA.Description}; % Display names of the scanned folders
{imgSetsP.Description}; % Display names of the scanned folders
{imgSetsT.Description}; % Display names of the scanned folders

%% Prepare Training and Validation Image Sets
% Since |imgSets| above contains an unequal number of images per category,
% let's first adjust it, so that the number of images in the training set is balanced.

%patientsPerLesion = [17, 33, 23, 11, 11];
patientsPerLesion = [2, 2, 2, 2, 2];
right = 0;
for lesionType = 1:5
    for patient = 1:patientsPerLesion(lesionType)
        disp([num2str(lesionType),' ',num2str(patient)])
    end
end

% lesionType = 2;
% patient = 7;

[imgSetsA, testA] = excludePatient(imgSetsA, lesionType, patient);
[imgSetsP, testP] = excludePatient(imgSetsP, lesionType, patient);
[imgSetsT, testT] = excludePatient(imgSetsT, lesionType, patient);

minSetCountA = min([imgSetsA.Count]); % determine the smallest amount of images in a category
minSetCountP = min([imgSetsP.Count]); % determine the smallest amount of images in a category
minSetCountT = min([imgSetsT.Count]); % determine the smallest amount of images in a category

%% Loop
count = 1;
NO_OF_LOOPS = 1
while (count <= NO_OF_LOOPS)
% Use partition method to trim the set.
%imgSetsA = partition(imgSetsA, minSetCountA, 'randomize');
%imgSetsP = partition(imgSetsP, minSetCountP, 'randomize');
%imgSetsT = partition(imgSetsT, minSetCountT, 'randomize');
imgSetsAPartition = partition(imgSetsA, minSetCountA, 'randomize');
imgSetsPPartition = partition(imgSetsP, minSetCountP, 'randomize');
imgSetsTPartition = partition(imgSetsT, minSetCountT, 'randomize');
% Notice that each set now has exactly the same number of images.
[imgSetsAPartition.Count]
[imgSetsPPartition.Count]
[imgSetsTPartition.Count]
%%
% Separate the sets into training and validation data. Pick x% of images
% from each set for the training data and the remainder, (100-x)%, for the 
% validation data. Randomize the split to avoid biasing the results.

% [trainingSetsA, validationSetsA] = partition(imgSetsA, 0.5, 'randomize');
% [trainingSetsP, validationSetsP] = partition(imgSetsP, 0.5, 'randomize');
% [trainingSetsT, validationSetsT] = partition(imgSetsT, 0.5, 'randomize');

%%
% Separate the sets into training and test data.
trainingSetsA = imgSetsAPartition;
trainingSetsP = imgSetsPPartition;
trainingSetsT = imgSetsTPartition;

testSetsA = imageSet(testA);
testSetsP = imageSet(testP);
testSetsT = imageSet(testT);

%% Create a Visual Vocabulary and Train an Image Category Classifier
% Bag of words is a technique adapted to computer vision from the
% world of natural language processing. Since images do not actually
% contain discrete words, we first construct a "vocabulary" of 
% <matlab:doc('extractFeatures'); SURF> features representative of each image category.

%%
% This is accomplished with a single call to |bagOfFeatures| function,
% which:
%
% # extracts SURF features from all images in all image categories
% # constructs the visual vocabulary by reducing the number of features
%   through quantization of feature space using K-means clustering
bagA = bagOfFeatures(trainingSetsA,'GridStep', [4, 4], 'BlockWidth',[32 64 96 128],'Upright' , false, 'VocabularySize', 500);
bagP = bagOfFeatures(trainingSetsP,'GridStep', [4, 4], 'BlockWidth',[32 64 96 128],'Upright' , false, 'VocabularySize', 500);
bagT = bagOfFeatures(trainingSetsT,'GridStep', [4, 4], 'BlockWidth',[32 64 96 128],'Upright' , false, 'VocabularySize', 500);
%%
% Additionally, the bagOfFeatures object provides an |encode| method for
% counting the visual word occurrences in an image. It produced a histogram
% that becomes a new and reduced representation of an image.
%%
% This histogram forms a basis for training a classifier and for the actual
% image classification. In essence, it encodes an image into a feature vector. 
%
% Encoded training images from each category are fed into a classifier
% training process invoked by the |trainImageCategoryClassifier| function.
% Note that this function relies on the multiclass linear SVM classifier
% from the Statistics and Machine Learning Toolbox(TM).

categoryClassifierA = trainImageCategoryClassifier(trainingSetsA, bagA);
categoryClassifierP = trainImageCategoryClassifier(trainingSetsP, bagP);
categoryClassifierT = trainImageCategoryClassifier(trainingSetsT, bagT);
%%
% The above function utilizes the |encode| method of the input |bag| object
% to formulate feature vectors representing each image category from the 
% |trainingSets| array of imageSet objects.

%% Evaluate Classifier Performance
% Now that we have a trained classifier, |categoryClassifier|, let's
% evaluate it. As a sanity check, let's first test it with the training
% set, which should produce near perfect confusion matrix, i.e. ones on 
% the diagonal.

disp('Evaluating the training set ...')
[confMattrA,knownLabelIdxtrA,predictedLabelIdxtrA,scoretrA]  = evaluate(categoryClassifierA, trainingSetsA);
[confMattrP,knownLabelIdxtrP,predictedLabelIdxtrP,scoretrP]  = evaluate(categoryClassifierP, trainingSetsP);
[confMattrT,knownLabelIdxtrT,predictedLabelIdxtrT,scoretrT]  = evaluate(categoryClassifierT, trainingSetsT);
disp('Evaluating the training set done!')
%%
% Next, let's evaluate the classifier on the validationSet, which was not
% used during the training. By default, the |evaluate| function returns the
% confusion matrix, which is a good initial indicator of how well the
% classifier is performing.

% disp('Evaluating the validation set ...')
% [confMatrixvA,knownLabelIdxvA,predictedLabelIdxvA,scorevA]  = evaluate(categoryClassifierA, validationSetsA);
% [confMatrixvP,knownLabelIdxvP,predictedLabelIdxvP,scorevP]  = evaluate(categoryClassifierP, validationSetsP);
% [confMatrixvT,knownLabelIdxvT,predictedLabelIdxvT,scorevT]  = evaluate(categoryClassifierT, validationSetsT);
% disp('Evaluating the validation set done!')

%% Compute average accuracy
% resultA(count) = mean(diag(confMatrixvA))
% resultP(count) = mean(diag(confMatrixvP))
% resultT(count) = mean(diag(confMatrixvT))

%% predict
% For all images of the same phase
% Convert imageSet to Datastore
imds = imageDatastore(testSetsA.ImageLocation);
[labelIdxA, scoreA] = predict(categoryClassifierA,imds)
% For a particular image
%[labelIdx, score] = predict(categoryClassifierA,imread(testSetsA.ImageLocation{1, 5}));
% Display the classification label.
categoryClassifierA.Labels(labelIdxA)
targetIdxA = repmat(lesionType,1, length(labelIdxA));
accuracyA(lesionType, patient) = sum(labelIdxA == targetIdxA')/numel(labelIdxA');

imds = imageDatastore(testSetsP.ImageLocation);
[labelIdxP, scoreP] = predict(categoryClassifierP,imds)
% For a particular image
%[labelIdx, score] = predict(categoryClassifierA,imread(testSetsA.ImageLocation{1, 5}));
% Display the classification label.
categoryClassifierP.Labels(labelIdxP)
targetIdxP = repmat(lesionType,1, length(labelIdxP));
accuracyP(lesionType, patient) = sum(labelIdxP == targetIdxP')/numel(labelIdxP');

imds = imageDatastore(testSetsT.ImageLocation);
[labelIdxT, scoreT] = predict(categoryClassifierT,imds)
% For a particular image
%[labelIdx, score] = predict(categoryClassifierA,imread(testSetsA.ImageLocation{1, 5}));
% Display the classification label.
categoryClassifierT.Labels(labelIdxT)
targetIdxT = repmat(lesionType,1, length(labelIdxT));
accuracyT(lesionType, patient) = sum(labelIdxT == targetIdxT')/numel(labelIdxT');

diagnostic = [labelIdxA; labelIdxP; labelIdxT]
%scores = [max(scoreA,[],2);max(scoreP,[],2);max(scoreT,[],2)]
[tmp,~,idx] = unique(diagnostic')
if tmp(mode(idx))==lesionType
    right = right + 1
end
accuracy = right*100/sum(patientsPerLesion)
%% Compute elapsed time
endTime = toc(startTime);
disp(['Time elapsed: ', num2str(endTime/60), ' mins'])

% accuracy = sum(YPred == YTest)/numel(YTest) 
% [tmp,~,idx] = unique(categoryClassifierA.Labels(labelIdxA)) 
% y3 = {'Poor','Good','Good','Goood'}; 
% [tmp,~,idx] = unique(y3); 
% tmp{mode(idx)}