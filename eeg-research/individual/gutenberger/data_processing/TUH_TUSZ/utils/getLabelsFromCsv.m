function [startStopTimes] = getLabelsFromCsv(filepath)
    csvLines = importdata(filepath, ',').textdata;
    read = false;
    startStopTimes = [];
    stopTimes = [];

    for l = 1:size(csvLines, 1)
        if (read && strcmp(csvLines{l,4}, 'seiz'))
            startStopTimes = [startStopTimes; [str2double(csvLines{l,2}), str2double(csvLines{l,3})]];
        end
        if(strcmp(csvLines{l,1}, 'channel')) 
            read = true;
        end



    end
end