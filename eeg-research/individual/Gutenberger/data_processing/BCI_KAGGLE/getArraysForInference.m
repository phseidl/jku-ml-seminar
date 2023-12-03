EEGOUT1 = pop_loadset( 'filename', 'Data_S18.set', 'filepath', 'bci-challenge/original');
EEGOUT2 = pop_loadset( 'filename', 'Data_S18_Sess.set', 'filepath', 'bci-challenge/ICA');

% pop_eegplot(EEGOUT1);
% pop_eegplot(EEGOUT2);

x_test = EEGOUT1.data(:, 1000:10000);
y_test = EEGOUT2.data(:, 1000:10000);

figure;
plot(x_test(1,:));
hold on;
plot(y_test(1,:));
legend('1', '2');

save('Data_S18.mat', 'x_test', 'y_test');