%% Lab 3 - Digitial Filters Design, Implementation and Analysis
clear variables;
close all;
format short;
%% Task 1-a FIR Filter Function

% Impulse Response
h = [0.2 0.3 0.2];
% Input Signal
x = [1 2 3 4 5 6 7 8 9 10];

% Initialise filter output array
y = zeros(1, length(x));

% test myFIR filter function
y = myFIR(x, h);
% Use matlab function as a comparison
y_m = filter(h, 1, x);

% Plot results
plot(x);        % original signal
hold on;
plot(y, 'r');   % myFIR
plot(y_m, 'ko');% Matlab filter

legend(["Original" "myFIR" "Matlab"]);
title("FIR filter implementation");
xlabel("Sample n");
%% Task 1-b Delta Impulse Sequence
close all;

% delta impulse sequence
d = zeros(1,10);
d(1) = 1;

% filter outputs with myFIR and matlab
y_d = myFIR(d,h);
y_dm = filter(h, 1, d);

% Plot results
plot(d);        % original signal
hold on;
plot(y_d, 'r');   % myFIR
plot(y_dm, 'ko');% Matlab filter

legend(["Original" "myFIR" "Matlab"]);
title("FIR filter implementation");
xlabel("Sample n");

%% Task 2-a Generating Sampled Signal
close all;
clear variables;

% 8kHz
fs = 8e3;
ts = 1/fs;
%sample for a second
t = 0:ts:1;

x = 5*cos(2*pi*500*t)+5*cos(2*pi*1200*t)+5*cos(2*pi*1800*t+0.5*pi);

plot(x(1:100));

%% Task 2-b Impulse Responses
close all;

b1 = [ -0.0012  -0.0025  -0.0045  -0.0068  -0.0073 ...
       -0.0030   0.0089   0.0297   0.0583   0.0907 ...
        0.1208   0.1422   0.1500   0.1422   0.1208 ...
        0.0907   0.0583   0.0297   0.0089  -0.0030 ...
       -0.0073  -0.0068  -0.0045  -0.0025  -0.0012];
   
b2 =[   0.0004  -0.0017  -0.0064  -0.0076   0.0073 ...
        0.0363   0.0458   0.0000  -0.0802  -0.1134 ...
       -0.0419   0.0860   0.1500   0.0860  -0.0419 ...
       -0.1134  -0.0802   0.0000   0.0458   0.0363 ...
        0.0073  -0.0076  -0.0064  -0.0017   0.0004 ];
    
subplot(2,1,1);
stem(b1);
subplot(2,1,2);
stem(b2);

%% Task 2-c Filtering and fourier transforms
close all;

y1 = myFIR(x, b1);
y2 = myFIR(x, b2);

x_fft =  abs(fft(x));
y1_fft = abs(fft(y1));
y2_fft = abs(fft(y2));

subplot(3,1,1);
plot(x_fft(1:4001));
title("Unfiltered Signal x");
xlabel("Frequency (Hz)");
subplot(3,1,2);
plot(y1_fft(1:4001));
title("Impulse Response h1");
xlabel("Frequency (Hz)");
subplot(3,1,3);
plot(y2_fft(1:4001));
title("Impulse Response h2");
xlabel("Frequency (Hz)");

%% Task 3 freqz
close all;
clear variables;

fs =8e3;

%filter 1
b1 = [ -0.0012  -0.0025  -0.0045  -0.0068  -0.0073 ...
       -0.0030   0.0089   0.0297   0.0583   0.0907 ...
        0.1208   0.1422   0.1500   0.1422   0.1208 ...
        0.0907   0.0583   0.0297   0.0089  -0.0030 ...
       -0.0073  -0.0068  -0.0045  -0.0025  -0.0012];
% filter 2   
b2 =[   0.0004  -0.0017  -0.0064  -0.0076   0.0073 ...
        0.0363   0.0458   0.0000  -0.0802  -0.1134 ...
       -0.0419   0.0860   0.1500   0.0860  -0.0419 ...
       -0.1134  -0.0802   0.0000   0.0458   0.0363 ...
        0.0073  -0.0076  -0.0064  -0.0017   0.0004 ];

% take frequency response for each filter
[h1, w1] = freqz(b1, 1, 512, fs);
[h2, w2] = freqz(b2, 1, 512, fs);


subplot(2,1,1);
plot(w1,20*log10(abs(h1)));
title("FIR Filter 1");
xlabel("Frequency (Hz)");
ylabel("Magnitude (dB)");
subplot(2,1,2);
plot(w2,20*log10(abs(h2)));
title("FIR Filter 2");
xlabel("Frequency (Hz)");
ylabel("Magnitude (dB)");

%% Task 4 we.dat
close all;
clear variables;

%filter 1
b1 = [ -0.0012  -0.0025  -0.0045  -0.0068  -0.0073 ...
       -0.0030   0.0089   0.0297   0.0583   0.0907 ...
        0.1208   0.1422   0.1500   0.1422   0.1208 ...
        0.0907   0.0583   0.0297   0.0089  -0.0030 ...
       -0.0073  -0.0068  -0.0045  -0.0025  -0.0012];
% filter 2   
b2 =[   0.0004  -0.0017  -0.0064  -0.0076   0.0073 ...
        0.0363   0.0458   0.0000  -0.0802  -0.1134 ...
       -0.0419   0.0860   0.1500   0.0860  -0.0419 ...
       -0.1134  -0.0802   0.0000   0.0458   0.0363 ...
        0.0073  -0.0076  -0.0064  -0.0017   0.0004 ];

% import we.dat file    
x = importdata('WE.DAT');

% filter x with b1 and b2
y1 = myFIR(x, b1);
y1_ = filter(b1, 1, x);
y2 = myFIR(x, b2);
y2_ = filter(b2, 1, x);

% calculate frequency spectrum for original and filtered signals
x_fft = abs(fft(x));
y1_fft = abs(fft(y1));
y2_fft = abs(fft(y2));

% x axis for spectrum plots
f = 0:4:4000;

% plot unfiltered and filtered spectrums for comparison
subplot(3,1,1);
plot(f, x_fft(1:1001));
title("Unfiltered Signal x");
xlabel("Frequency (Hz)");
subplot(3,1,2);
plot(f, y1_fft(1:1001));
title("Impulse Response b1");
xlabel("Frequency (Hz)");
subplot(3,1,3);
plot(f, y2_fft(1:1001));
title("Impulse Response b2");
xlabel("Frequency (Hz)");

%% Task 5 Low Pass FIlter
close all;
clear variables;

fs =44100;
lp = designfilt('lowpassfir', ...
                'SampleRate', fs, ...
                'StopBandAttenuation', 50, ...
                'StopBandFrequency', 1400, ...
                'PassbandFrequency', 600, ...
                'PassbandRipple', 0.02);

hp = designfilt('highpassfir', ...
                'SampleRate', fs, ...
                'StopBandAttenuation', 50, ...
                'StopBandFrequency', 600, ...
                'PassbandFrequency', 1400, ...
                'PassbandRipple', 0.02);

fvtool(lp);

hold on;
            
fvtool(hp);


            
%% Task 6-a FIR Notch Filter design
close all;
clear variables;

%import data and calculate number of samples
ecg = importdata('ecgbn.dat');
N = length(ecg);

%sample frequency
fs = 600;
%notch filter centre frequencies
f1 = 60;
f2 = 120;
f3 = 180;

%Calculate the filter coefficients for each filter
%Filter 1 60Hz
b1 = myFirNotch(fs, f1);
% Filter 2 - 120Hz
b2 = myFirNotch(fs, f2);
% Filter 3 - 180Hz
b3 = myFirNotch(fs, f3);

%Plot frequency response of each filter
fvtool(b1,1,b2,1,b3,1, 'Fs', fs);
legend("60Hz","120Hz", "180Hz");

%% Task 6-b Comnining filters and comparing signal spectrums
close all;

%Cascade the 60Hz, 120Hz, and 180Hz notch filters
ecg1 = myFIR(ecg, b1);  %60
ecg2 = myFIR(ecg1, b2); %120
ecg3 = myFIR(ecg2, b3); %180

%plot signal pre and post filtering
figure(1);
subplot(2,1,1);
plot(ecg);
title("Noisy ecg signal");
subplot(2,1,2);
plot(ecg3);
title("Filtered ecg signal");

%plot frequency spectrum of signal before and after filtering
df = fs/N;
f = -fs/2:df:fs/2-df;
figure(2);
subplot(2,1,1);
plot(f, abs(fftshift(fft(ecg))));
title("Spectrum before filtering");
xlabel("Frequency (Hz)");
subplot(2,1,2);
plot(f, abs(fftshift(fft(ecg3))));
title("Spectrum after filtering");
xlabel("Frequency (Hz)");


%% Task 7 - IIR Filter frequency response for different r values
close all;
clear variables;


theta = pi/3;
r = [0.99, 0.9, 0.8];

% generate filter coefficients for all r values
for filter = 1:3
    b(filter)    = 1-r(filter);
    a(filter, 1) = 1;
    a(filter, 2) = -2*r(filter)*cos(theta);
    a(filter, 3) = r(filter)^2;
end

% plot the frequency response for each filter
fvtool(b(1), a(1, :), b(2), a(2,:), b(3), a(3, :));
legend('r=0.99', 'r=0.9', 'r=0.8');
% r is inversely proportioanal to the bandwidth of the filter. As r
% increases the peak narrows.

%% Task 8 - IIR Filter

theta = [pi/6, pi/3, pi/2];
r   = 0.9;

% generate filter coefficients for all r values
for filter = 1:3
    b(filter)    = 1-r;
    a(filter, 1) = 1;
    a(filter, 2) = -2*r*cos(theta(filter));
    a(filter, 3) = r^2;
end

% plot the frequency response for each filter
fvtool(b(1), a(1, :), b(2), a(2,:), b(3), a(3, :));
legend('phi=pi/6', 'phi=pi/3', 'phi=pi/2');

% theta is directly proportional to the location of the peak. As it
% increases the centre frequency increases

%% Task 8
close all;
clear variables;

theta = [pi/2, pi/3, pi/6];
r = 0.9;

% generate filter coefficients for all r values
for filter = 1:3
    b(filter)    = 1-r;
    a(filter, 1) = 1;
    a(filter, 2) = -2*r*cos(theta(filter));
    a(filter, 3) = r^2;
end

% plot the frequency response for each filter
fvtool(b(1), a(1, :), b(2), a(2,:), b(3), a(3, :));
legend('theta=pi/2', 'theta=pi/3', 'theta=pi/6');
% r is inversely proportioanal to the bandwidth of the filter. As r
% increases the peak narrows.


%% Task 9 - 
close all;
clear variables;


pcm = importdata('pcm.mat');

fs = 8192;
f = (-1:(2/15999):1);

subplot(2,1,1);
plot(pcm)
subplot(2,1,2);
plot(f,abs(fftshift(fft(pcm))));
xticks([-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]);
xlim([-1 1]);
xlabel('Normalised \pi rads/sample');
set(gca, 'YAxisLocation', 'origin');
sound(pcm);

% From the spectrum plot, the narrowband frequency is centred around 0.7866pi rads
% The bandwidth is around 0.008pi rads


%%
close all;
% narrow band frequency peak at normalized frequency of 0.7866pi rad
% band width of ~0.110pi rads
theta = 0.7866*pi;
r   = 1-(0.008*pi/2);

% generate filter coefficients for all r values
b   = 1-r;
a(1) = 1;
a(2) = -2*r*cos(theta);
a(3) = r^2;

%poles at r*e^j*om and r*e^-j*om
%using euler r*e^j*om becomes j*om
%p1 = j*om, p2 = -j*om
%z1 = 1,    z2 = -1
%transfer function (z-z1)(z-z2)/(z-p1)(z-p2)
%expanding brackets gives (z^2-1)/(z^2+
%a = [1, 2*r*cos(om_o), r^2];
%b = [1, 0, -1];

fvtool(b,a, 'Fs', fs);
subplot(2,1,1);
plot(f,abs(fftshift(fft(pcm))));
pcm_filt = filter(b,a,pcm);
subplot(2,1,2);
plot(f, abs(fftshift(fft(pcm_filt))));
set(gca, 'YAxisLocation', 'origin');
