function [OUTEEG] = removeChannels(keepChannelNames,EEG)
    removeIdx = linspace(1, EEG.nbchan, EEG.nbchan);
   
    for i = 1 : EEG.nbchan
        for j = 1 : size(keepChannelNames, 2)
            if (contains(EEG.chanlocs(i).labels, strcat(keepChannelNames{j}, '-LE')) || contains(EEG.chanlocs(i).labels,strcat(keepChannelNames{j}, '-REF')))
                removeIdx(i) = 0;
                
            end
        end
    end
    
    OUTEEG = pop_select(EEG, 'rmchannel', nonzeros(removeIdx));

end