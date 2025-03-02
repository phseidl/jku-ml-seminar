function [EEG] = removeNonBrainComponents(EEG)
    
    flag=zeros(size(EEG.icaweights,1)); % let component remain or remove

    data_projected = EEG.icaweights*EEG.icasphere*EEG.data;
    inv=EEG.icawinv;
    
    for i =1:size(EEG.icaweights,1) % check which components is classify to brain type
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
    
    for i = size(EEG.icaweights,1):-1:1 % remove non brain components
        if(flag(i)==0)
            data_projected(i,:)=[];
            inv(:,i)=[];
        end
    end
    
    final_data=inv*data_projected;
    EEG.data=final_data;

    EEG = pop_editset(EEG, 'data',  final_data);    % add brain only data to EEG struct
    
    end