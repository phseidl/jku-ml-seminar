% Dataset "inria-bci-challenge" is available on (https://www.kaggle.com/c/inria-bci-challenge)
subject=['16','17','18','20','21','22','23','24','26'];

labelind=1; %count for subject label

if ~exist('bci-challenge/epoch/test', 'dir')
    mkdir('bci-challenge/epoch/test');
end

if ~exist('bci-challenge/epoch/val', 'dir')
    mkdir('bci-challenge/epoch/val');
end

for num = 1:2:length(subject)
    foldername=['test'];
    save_datapath = ['bci-challenge/epoch/',foldername,'/'];
    alllabel = csvread('inria-bci-challenge/TrainLabels.csv', 1, 1);
    allEpochData = [];
    flag=zeros(56);% let component remain or remove
    EpochData = [];
    EEG = [];
    %cd(EEGLAB_path);
    eeglab;
    EEG = eeg_checkset( EEG );
    eegfilenames = ['Data_S',subject(num:num+1),'.set'];
    EEG = pop_loadset('filename',eegfilenames,'filepath','bci-challenge/ICA/');
    data_projected = EEG.icaweights*EEG.icasphere*EEG.data;
    inv=EEG.icawinv;

    for i =1:55 %check which components is classify to brain type
        max=EEG.etc.ic_classification.ICLabel.classifications(i,1);
        for j =1:7
            if(EEG.etc.ic_classification.ICLabel.classifications(i,j)>max)
                max=EEG.etc.ic_classification.ICLabel.classifications(i,j);
            end
        end
        if(max==EEG.etc.ic_classification.ICLabel.classifications(i,1))
            flag(i)=1;
            
        end 
    end
    
    for i =55:-1:1 %remove others types of component
        if(flag(i)==0)
            data_projected(i,:)=[];
            inv(:,i)=[];
        end
    end
    
    final_data=inv*data_projected;
    EEG.data=final_data;

    ChList = [1:56]; % select all 56 EEG channels
    trial_c = 0; % trial count
    temp = [];
    for i_event = 1:size(EEG.event,2)
        temp.TrialOnset = EEG.event(1,i_event).latency; % obtain at which data point (when) the trial starts
        trial_c = trial_c + 1;
        TrialTiming.TrialOnset(trial_c,:) = temp.TrialOnset; % obtain the time of erp onsets
        temp = [];
        % extract the epoch from +0s to 1.25s of the trial start
        EpochData(trial_c,:,:) = double(EEG.data(ChList,ceil([(0+(1/EEG.srate)):1/EEG.srate:1.25]*EEG.srate)+floor(TrialTiming.TrialOnset(trial_c,1))));
    end

    allEpochData=EpochData;

    label = alllabel(labelind:labelind+339,:);
    x_test = allEpochData(:,:,:);
    y_test = label;
    savefilename = ['Data_S',subject(num:num+1),'_Sess.set'];
    %save([save_datapath,savefilename],'EEG');
    labelind=labelind+340;
    
    %saveset=[filename,'.set'];
	EEG = pop_saveset(EEG, 'filename',savefilename,'filepath','bci-challenge/ICA/'); % save data
end

