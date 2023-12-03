function process_TUH_data(startsub,stopsub,montage, dataRoot,saveroot, chanLocsPath, channels, removeSeiz, ica, saveArrays)

% process_TUH_data  Processes the TUSZ corpus of the TUH EEG dataset.
%   Dataset available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
%
%   Dataset is processed from startsub to stopsub 
% 
%   ARGUMENTS:
%   startsub    ...     [int] first subject index to process data
%   stopsub     ...     [int] last subject index to process data
%   montage     ...     [string] montage that should be processed, either
%   '01_tcp_ar', '02_tcp_le', '03_tcp_ar_a', '04_tcp_le'
%   dataRoot    ...     [string] root directory to the dataset (data structure should follow the original one from TUSZ!)
%                       e.g. dataRoot = "C:/TUHDATASET/home/edf/dev"
%   saveroot    ...     [string] directory where the .set files should be
%                       saved
%   channels    ...     [cell] cell array of strings with channel names to
%                       keep
%                       e.g. channels = channels = {'FP1', 'FP2', 'F3', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'};
%   removeSeiz  ...     [bool] defines if time intervals labelled as
%                       seizures should be removed before being stored into .set files
%   ica         ...     [bool] defines if ICA + ICLabel should be executed
%                       (.set files of IC + ICLabel are stored in a
%                       seperate ICA folder within saveroot)
%   saveArrays  ...     [bool] defines if labeled data arrays should be stored as .mat
%                       files
%
%   OUTPUT:
%   .set files are saved to saveroot 
%   There is one .set file for each patient, the sessions and recordings
%   that correspond to the selected montage are merged.
%   Naming convention of .set file: 'Data_{PATIENT_ID}.set'
%   

    EEG = [];
    
    % if they don't exist, create destination folders for dataset and ICA data
    if ~exist(string(saveroot) + '/original/', 'dir')
        mkdir(string(saveroot)  + '/original/');
    end
    
    if ~exist(string(saveroot)  + '/ICA/', 'dir')
        mkdir(string(saveroot)  + '/ICA/');
    end
    
    if saveArrays && ~exist(string(saveroot)  + '/arrays/', 'dir') 
        mkdir(string(saveroot)  + '/arrays/');
    end
    
    eeglab
    path = fullfile(dataRoot);
    [f,d] = getContent(path, 1);

    for iPatient = startsub:stopsub
        saveFilename = 'DataArray_S' + string(sprintf('%03i', iPatient)) + '.mat';
        saveFilenameSet = 'Data_S' + string(sprintf('%03i', iPatient)) + '.set';
        saveFilenameICASet = 'Data_S' + string(sprintf('%03i', iPatient)) + '_ICA.set';
        data = [];
        labels = [];
        p = fullfile(f{iPatient}, d{iPatient});
        subjectstrname = d{iPatient};
        [f2,d2] = getContent(p, 1);
        N2 = size(d2,1);   %number of sessions
        for iSession = 1:N2
            pf = fullfile(f2{iSession}, d2{iSession});
            foldername = d2{iSession};
            sessionname = strsplit(foldername,'_');
            sessionname = sessionname{1};
            [f3,montageSess] = getContent(pf, 1);
            pf = fullfile(f3{1}, montageSess{1});
            if(strcmp(montageSess{1}, montage))
                recnames = findRecording(pf, subjectstrname, sessionname);
                for irec = 1:size(recnames,1)
                    fprintf('processing ifolder: %d, isubject: %d, isession: %d, irec: %d \n', iPatient, iPatient, iSession, irec);
                    display([pf, ' -> ', recnames.recstrname{irec}]);

                    % read edf file
                    edfPath = fullfile(pf, recnames.edfname{irec});
                    EEG_temp = pop_biosig(edfPath,'importevent','off','rmeventchan','off');
                    EEG_temp = removeChannels(channels,EEG_temp);   %remove unwanted channels
                    
                    % rename channels to correspond to chanloc path (remove
                    % '-LE' or '-REF')
                    for i = 1 : EEG_temp.nbchan
                        newChLabel = erase(EEG_temp.chanlocs(i).labels,'-LE'); 
                        newChLabel = erase(newChLabel,'-REF');
                        EEG_temp = pop_chanedit(EEG_temp, 'changefield',{i,'labels', newChLabel});
                    end

                    EEG = [EEG; EEG_temp];

                    % read csv_bi file containing start and stop times of
                    % seizures
                    csvPath = fullfile(pf, recnames.csvbiname{irec});
                    seizureTimes = getLabelsFromCsv(csvPath);
                    
                    % create label vector, 1 = seizure, 0 = no seizure
                    labels = zeros(1, EEG_temp.pnts);
                    for i = 1:size(seizureTimes,1)
               
                        startIdx = round(seizureTimes(i,1)*EEG_temp.srate);
                        stopIdx = round(seizureTimes(i,2)*EEG_temp.srate);
                        
                        if (startIdx == 0)
                            startIdx = 1;
                        end
                        if (stopIdx > EEG_temp.pnts)
                            stopIdx = EEG_temp.pnts;
                        end

                        labels(startIdx:stopIdx) = 1;    %from start to end of seizure, label = 1
                        
                        if (removeSeiz)
                            seizurefreeData = EEG_temp.data;
                            seizurefreeData(:,startIdx:stopIdx) = [];
                            EEG_temp = pop_editset( EEG_temp, 'data',  seizurefreeData);
                        end
                    end
                    
                    if (saveArrays)
                        dataLabeled = [EEG.data; labels];
                        data = [data, dataLabeled];
                    end
                end
            end
        end
        if (saveArrays)
            save(string(saveroot) + '/arrays/' + saveFilename, 'data');
        end
        mergeIndices = 1:1:size(EEG,1);
        OUTEEG = pop_mergeset(EEG, mergeIndices, 0);

        OUTEEG = pop_resample( OUTEEG, 128); % downsampling to 128Hz
        OUTEEG = eeg_checkset( OUTEEG );
        OUTEEG = pop_reref( OUTEEG, []); % re-reference by CAR
        OUTEEG = eeg_checkset( OUTEEG );
        OUTEEG = pop_eegfiltnew(OUTEEG, 'locutoff',1,'hicutoff',50,'plotfreqz',1); % Band pass filter to 1-40 Hz
        OUTEEG = eeg_checkset( OUTEEG );


        OUTEEG = pop_chanedit(OUTEEG, 'lookup', chanLocsPath);
        
	    OUTEEG = pop_saveset( OUTEEG, 'filename', char(saveFilenameSet),'filepath', [saveroot, '/original/']); % save data
	
        % ICA and ICLabel of merged patients' EEG data
        if (ica)
            tic;
            OUTEEG = pop_runica(OUTEEG, 'extended',1,'interupt','on'); % run ica
            icaTime = toc;
            fprintf('ICA for patient %d finished after %d seconds (number of time points of the merged recording: %d). \n', iPatient, icaTime, OUTEEG.pnts);
                   
            OUTEEG = pop_iclabel(OUTEEG, 'default'); % run iclabel
            OUTEEG = removeNonBrainComponents(OUTEEG);
            OUTEEG = pop_saveset( OUTEEG, 'filename', char(saveFilenameICASet),'filepath', [saveroot, '/ICA/']); % save data

        end
    end


end