eeglab nogui


fileAppend='1B_2';
fftw('planner','measure');
% load singlewisdom
% temp=fftw('swisdom',singlewisdom);


dataFolder='C:\Users\au207178\OneDrive - Aarhus universitet\forskning\sleepInOrbit\sleepInOrbitPilots\sub-001\ses-002';
targetFolder='C:\Users\au207178\OneDrive - Aarhus universitet\forskning\sleepInOrbit\sleepInOrbitPilots\derivatives\cleaned_1\sub-001\ses-002';

%%
% eeglab
% records(iRec).recordName

disp('Loading')
EEG= pop_loadset(fullfile(dataFolder,'20230103_KM_søvn.set'));

EEG.data([3,8],:)=nan;

for iCh = 1:EEG.nbchan
    EEG.data(iCh,:) = EEG.data(iCh,:) - nanmean(EEG.data(iCh,:));  % remove channel mean
end

EEG = pop_eegfiltnew(EEG, .3, 100); % filter the data

EEG=pop_resample(EEG,250);

%     EEG=pop_select(EEG,'time',[0 1500*EEG.srate]);

fiftyHzPower=zeros(1,EEG.nbchan);

disp('Measuring 50 Hz')
%we measure 50 Hz contribution, relative to the power in the 5-51 Hz
%band (so it is bounded by 1), before notch-filtering it out later.
for iChan=1:EEG.nbchan
    nansamples=isnan(EEG.data);
    EEG.data(nansamples)=0;

    [Pxx,F] = pwelch(EEG.data(iChan,:),[],[],[],EEG.srate);
    fiftyHzPower(iChan)=real(sum(Pxx(F>49.5 & F < 50.5))/sum(Pxx(F>5 & F < 51)));

    EEG.data(nansamples)=nan;
end

%     load(fullfile(records(iRec).tempFolder,'compromisedChannels.mat'))

compromised=zeros(13,1); %compromised(compromisedChannels)=1;

%% Saturated channels

%saturated channels (out of range) are marked by the amplifier, so we
%can extract that information immediately: (note that whether saturated
%samples are nan, 0, or -52 V depends on how the recording is loaded into matlab).


saturated=sum(isnan(EEG.data'));


backupSaturation=isnan(EEG.data);

%% compromized channels



%%

%notch filtering (removing 50 and 100 Hz contributions):
disp('Notch Filtering')
EEG = pop_eegfiltnew(EEG,  49,51,846,1);
EEG = pop_eegfiltnew(EEG,  99,101,846,1);

%% remove single spikes
% in this step, a subfunction is called to detect short, large
% deviations found in a single electrode. If the deviation happens in
% multiple channels simultaneously, it is ignored

disp('Single Spikes')
allMean=nanmean(EEG.data);
EEG.data=EEG.data-allMean(ones(EEG.nbchan,1),:);

tempData=EEG.data;
EEG=electrode_spike_fun2(EEG);

[i,j]=find(~isnan(tempData) & isnan(EEG.data));
[m, n]=size(tempData);
vals=double(tempData(sub2ind([m n],i,j)));
spikeRemoval=sparse(i,j,vals,m,n);


% how much is gone now
removed_single_spikes=sum(isnan(EEG.data'))-saturated;

%% find bad electrodes
% here we focus on single electrodes with problematic behavior over long time, also detected in a subfunction

disp('Bad Electrodes')
tempData=EEG.data;
[EEG,medianLargeAmplitudes]=electrode_discard_fun5(EEG,1);

[i,j]=find(~isnan(tempData) & isnan(EEG.data));
[m, n]=size(tempData);
vals=double(tempData(sub2ind([m n],i,j)));
electrodeRemoval=sparse(i,j,vals,m,n);

% how much is now gone
removed_electrodes=sum(isnan(EEG.data'))-saturated-removed_single_spikes;

allMean=nanmean(EEG.data);
EEG.data=EEG.data-allMean(ones(EEG.nbchan,1),:);




%% remove movement
% as a last step, we detect large deviations across multiple channels
% simultaneously, which indicates either muscle activity or movement.
disp('Remove Movement epochs')

allMean=nanmean(EEG.data);
EEG.data=EEG.data-allMean(ones(EEG.nbchan,1),:);
tempData=EEG.data;

[EEG,~,EMGsamples]=EMGdetectFun3(EEG,1);

[i,j]=find(~isnan(tempData) & isnan(EEG.data));
[m, n]=size(tempData);
vals=double(tempData(sub2ind([m n],i,j)));
movementRemoval=sparse(i,j,vals,m,n);

% save(fullfile(records(iRec).tempFolder,['EMGsamples' fileAppend '.mat']),'EMGsamples')

% how much movement removed
removed_movement=sum(isnan(EEG.data'))-saturated-removed_single_spikes-removed_electrodes;




%%
% correct saturation for earPiece removal:
compCorrected=backupSaturation;
% %     compCorrected(compromisedChannels,:)=1;

leftPlugRemoved=all(compCorrected(1:4,:)==1);
leftPlugRemoved=leftPlugRemoved(ones(1,4),:);
rightPlugRemoved=all(compCorrected(5:8,:)==1);
rightPlugRemoved=rightPlugRemoved(ones(1,4),:);


saturated_activeEarpiece=min([sum((compCorrected(1:8,:))&~[leftPlugRemoved;rightPlugRemoved],2),saturated(1:8)'],[],2);

%% clean up remaining outliers
% we find any remaining large amplitude samples which have not been
% removed previously (such as simultanous large deviations in multiple
% channels for short periods of time). there is usually very little of
% this
allMean=nanmean(EEG.data);
EEG.data=EEG.data-allMean(ones(EEG.nbchan,1),:);

tempData=EEG.data;
EEG=electrode_spike_fun1(EEG);

[i,j]=find(~isnan(tempData) & isnan(EEG.data));
[m, n]=size(tempData);
vals=double(tempData(sub2ind([m n],i,j)));
finalRemoval=sparse(i,j,vals,m,n);

removed_final=sum(isnan(EEG.data'))-saturated-removed_single_spikes-removed_electrodes-removed_movement;


%% stats



% rejectStats=[saturated(1:13)' removed_single_spikes(1:13)'...
%     removed_electrodes(1:13)' removed_movement(1:13)' removed_final(1:13)' ]/EEG.pnts;
% saturated_activeEarpiece=saturated_activeEarpiece/EEG.pnts;

% save(fullfile(records(iRec).tempFolder,['rejectStats' fileAppend]),'rejectStats','saturated_activeEarpiece')
save(fullfile(targetFolder,['EEGcleaned' fileAppend ]),'EEG','-v7.3')
pop_saveset(EEG,fullfile(targetFolder,['EEGcleaned' fileAppend ]))

%% figure
ChannelLabels={EEG.chanlocs.labels};




figure('position',[        1000         127         727        1211])

subplot(24,1,1:3)
load(fullfile(records(iRec).tempFolder,'DC_levels'))

load(fullfile(records(iRec).tempFolder,'compromisedChannels.mat')   )

boxplot(DC')
set(gca,'xtick',1:13,'xticklabel',ChannelLabels,'XTickLabelRotation',90,'xlim',[0 14])

title('Electrode DC levels','fontsize',15)


text(.5,2.35,datestr(records(iRec).dateTime),'fontsize',25,'units','normalized','HorizontalAlignment','center')
ylabel('DC (uV)')

subplot(24,1,5:8)

threeThresholds=[5 1 0.04];%
origIndxs=[smoothEpochs(:) correlationStatsSimilar(:) largestBin(:)];
scaleIndxs=origIndxs./threeThresholds(ones(numel(compromisedChannels),1),:);    %rescaling according to thresholds

bar(1:13,scaleIndxs,.7)


hold on
plot([0 14],[1 1],'--r')
set(gca,'ytick',1,'yticklabel',{'Rejection'})

set(gca,'xtick',1:13,'xticklabel',ChannelLabels,'XTickLabelRotation',90)


title('Compromised Shielding','fontsize',15)
lh=legend('Smooth Epochs','Similar Channels','Edge of range','orientation','horizontal','location','southoutside');



subplot(24,1,10:15)

fun1=@(x,y) bar(x,y*100,.5,'stacked');
fun2=@(x,y) bar(x,y,.1);
ax=plotyy((1:EEG.nbchan)-.3,rejectStats,(1:EEG.nbchan)+.15,fiftyHzPower,fun1,fun2);

ylabel(ax(1),'Rejection (%)')
ylabel(ax(2),'50 Hz Power rel. to 5-51 Hz')


set(ax(1),'xtick',1:13,'xticklabel',ChannelLabels,'XTickLabelRotation',90,'ylim',[0 100],'ytick',0:20:100,'xlim',[0 14])
set(ax(2),'xlim',[0 14],'ylim',[0 .5],'ytick',[0:.1:.5])
legend('Saturated','Electrode spike','Bad electrode','Movement','Other','orientation','horizontal','location','southoutside')
title('Rejection','fontsize',15)





subplot(24,1,17:24)
hold on
for iC=1:EEG.nbchan
    [runstarts, rundur]=findRuns(isnan(EEG.data(iC,:)));
    tooShort=rundur<=EEG.srate*10;
    rundur(tooShort)=[]; runstarts(tooShort)=[];
    plot([runstarts;runstarts+rundur]/EEG.srate/3600,ones(2,numel(runstarts))*iC,'.-r','linewidth',1)
end

set(gca,'ylim',[0 EEG.nbchan+1],'ydir','reverse')
xlabel('Hrs')
ylabel('# Channel')
title('Fails over 10 seconds','fontsize',15)
set(gcf,'position',[1000          38        1114        1300]);

print(fullfile(records(iRec).tempFolder,['dataQualReport' fileAppend '.png']),'-dpng')
drawnow

singlewisdom = fftw('swisdom');
save singlewisdom singlewisdom


return
%%
% data3=data2;
% data3(~isnan(EEG.data))=nan;
data3=full(electrodeRemoval+finalRemoval+movementRemoval+spikeRemoval);
data3(data3==0)=nan;
eegplot(EEG.data,'data2',data3,'srate',EEG.srate,'events',EEG.event,'winlength',100,'spacing',100,'eloc_file', EEG.chanlocs )
