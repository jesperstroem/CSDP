%
% kald det her for de to pilotmålinger, men sørg for ikke at overskrive de eksisterende. sammenlign outputs'ne


dataSetFolder='C:\Users\au207178\OneDrive - Aarhus universitet\forskning\sleepInOrbit\sleepInOrbitPilots\';
bidsDir=fullfile(dataSetFolder,'derivatives','cleaned_1','sub-001');
targetDir=fullfile(dataSetFolder,'derivatives','spectrograms2_matlab','sub-001');

%1
cleanName=fullfile(bidsDir,'ses-001','EEGcleaned1B_2');
targetName=fullfile(targetDir,'ses-001');
prepareDataSpectrograms(cleanName,targetName,1:4,5:8)


%2
cleanName=fullfile(bidsDir,'ses-002','EEGcleaned1B_2');
targetName=fullfile(targetDir,'ses-002');
prepareDataSpectrograms(cleanName,targetName,1:4,5:8)

%%

function prepareDataSpectrograms(cleanDataPath, derivPath,Lidxs,Ridxs)
%special version that removes epochs where any of L,R,LR2 are missing.



if(~exist(derivPath, 'dir'))
    mkdir(derivPath);
end

temp=load(cleanDataPath);
[LR,freqs,usedEpochs] = process_one_file(temp.EEG,Lidxs,Ridxs);


X = single(LR);
X(isnan(X) | isinf(X))=0;
save([derivPath, '/eeg_lr.mat'], 'X', 'usedEpochs','freqs', '-v7.3');

end

function  [LR,freqs, usedEpochs] =...
    process_one_file(rawData,leftChans,rightChans)

%resample:
fs = 100;       % sampling frequency
rawData=pop_resample(rawData,fs);


data=rawData.data;
epochLength=fs*30;

nEpochs=floor(size(data,2)/epochLength);
data=data(:,1:nEpochs*epochLength);

%remove single-ear epochs:
usedEpochs=1:nEpochs;

% dummy=1:size(data,2);
% dummy=reshape(dummy,[epochLength,nEpoch]);
% % dummy(:,singleEarEpochs)=[];
% dummy=dummy(:)';
% data=data(:,dummy);
%
% usedEpochs(singleEarEpochs)=[];


% somehow some 'data' variables are saved in single format,
% causing trouble with resample, I convert them to double here
data = double(data);



% left-right derivation
LR = nanmean(data(leftChans,:)) - nanmean(data(rightChans,:));
LRnans=isnan(LR);
LR=interpolateOverNans(LR,fs);
if(sum(isnan(LR)) > 0)
    disp(['WARNING: NaNs found in LR derivation']);
end
[LR, freqs]= raw_to_stft(LR, fs);


assert(size(LR,1) == nEpochs);


%skip epochs where any channel is missing
missing=all(reshape(LRnans,[30*fs nEpochs]));


LR(missing,:,:) = [];
usedEpochs(missing) = [];



end

function [X, freqs] = raw_to_stft(raw_signal, fs)
% Short time fourier transform information
epoch_size = 30;
win_size  = 2;      % window size for STFT
overlap = 1;        % overlap size for STFT
nfft = 2^nextpow2(win_size*fs); % NFFT

epochs = buffer(raw_signal, epoch_size*fs);
epochs = epochs';
% remove the trailing epoch
if(mod(length(raw_signal), epoch_size*fs) > 0)
    epochs(end, :) = [];
end

N = size(epochs, 1);
X = zeros(N, 29, nfft/2+1);
for i = 1 : N
    [Xi,freqs,~] = spectrogram(epochs(i,:), hamming(win_size*fs), overlap*fs, nfft,fs);

    % log magnitude spectrum
    Xi = 20*log10(abs(Xi));
    Xi = Xi';
    X(i,:,:) = Xi;
end
end

function dataInterp=interpolateOverNans(allDeriv,fs)

%we can't have nans at the end:
allDeriv(isnan(allDeriv(:,1)),1)=1;
allDeriv(isnan(allDeriv(:,end)),end)=-1;


for iDeriv=1:size(allDeriv,1)



    nanSamples=find(isnan(allDeriv(iDeriv,:)) | isinf(allDeriv(iDeriv,:)));
    if numel(nanSamples)>0
        [nanStart, nanDur]=findRuns(isnan(allDeriv(iDeriv,:)));
        nanDur=nanDur-1;
        realSamples=unique([nanStart-1 (nanStart+nanDur)+1]);
        counter=0;


        distanceToReal=nanSamples*0;

        for iRun=1:numel(nanDur)
            distanceToReal(counter+(1:nanDur(iRun)))=[0:(floor(nanDur(iRun)/2)-1) (ceil(nanDur(iRun)/2)-1):-1:0 ];
            counter=counter+nanDur(iRun);
        end
        assert(numel(distanceToReal)==numel(nanSamples))

        interpValues=interp1(realSamples,allDeriv(iDeriv,realSamples),nanSamples,'linear');
        interpValues=interpValues.*exp(-distanceToReal/(fs*1));


        allDeriv(iDeriv,nanSamples)=interpValues;
    end

end

dataInterp=allDeriv;
end

function [runStarts, runLengths]=findRuns(sequence)
% [runStarts runLengths]=findRuns(sequence)

assert(all(sequence==0 | sequence==1),'Sequence is all 0 or all 1')

changes=diff([0 sequence 0]);
runStarts=find(changes>0);
runEnds=find(changes<0);
runLengths=runEnds-runStarts;
assert(all(runLengths>0))
end