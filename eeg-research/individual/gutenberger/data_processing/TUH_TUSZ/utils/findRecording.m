function [recs] = findRecording(root, subjectstrname, sessionname)
extEDF = 'edf';
% extCSV_bi = 'bi';
[f,d] = getContent(root, 0);
N = size(d,1);
recnames=[];
r=[];
for i = 1:N
    ss = strsplit(d{i},{'_','.'});
    if(strcmp(ss{1}, subjectstrname))
        if(strcmp(ss{2}, sessionname))
            % if(strcmp(char(ss(end)), extCSV_bi))
            %     r.csvbiname = d(i);
            %     csvnames = [recnames; r];
            % end
            if(strcmp(char(ss(end)), extEDF))
                r.recstrname = ss(3);
                r.recnum = str2num(r.recstrname{1}(2:end));
                r.edfname = d(i);
                r.csvbiname = {[subjectstrname,'_',sessionname,'_',r.recstrname{1},'.csv_bi']};
                recnames=[recnames; r];
            end
        end
    end
end
recs = struct2table(recnames);
recs = sortrows(recs, 'recnum');
end