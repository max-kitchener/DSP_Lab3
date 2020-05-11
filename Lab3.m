%% Lab 3 - Digitial Filters Design, Implementation and Analysis
% Task 1-a FIR Filter Function
clear variables;
close all;
format short;


% Impulse Response
h = [0.2 0.3 0.2];
% Input Signal
x = [1 2 3 4 5 6 7 8 9 10];

% Initialise filter output
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

legend(["Original" "myFIR" "Matlab"], 'Location', 'NorthWest');
title("FIR filter implementation");
xlabel("Sample n");

%Difference equation
%y(n) = 0.2x(n)+0.3x(n-1)+0.2x(n-2)
%Z-Transform
%Y(z) = 0.2X(z)+0/3X(z)z^-1+0.2X(z)z^-2
%Y(z) = X(z)(0.2+0.3z^-1+0.2z^-2)
%Transfer function
%Y(z)/X(z)=H(z)
%H(z) = 0.2+0.3z^-1+0.2z^-2

%% Task 1-b Delta Impulse Sequence
close all;

% delta impulse sequence
d = zeros(1,10);
d(1) = 1;

% filter outputs with myFIR and matlab
y_d = myFIR(d,h);
y_dm = filter(h, 1, d);

% Plot results
stem(d);        % original signal
hold on;
plot(y_d, 'r');   % myFIR
plot(y_dm, 'ko');% Matlab filter

legend(["Original" "myFIR" "Matlab"]);
title("FIR filter implementation");
xlabel("Sample n");

%output matches the filter coefficients,this can be predicted without
%calculation, as only x(0)=1 for an impulse response.
% y(0) = 0.2*1 + 0     + 0
% y(1) = 0     + 1*0.3 + 0
% y(2) = 0     + 0     + 1*0.2
% y(3) = 0     + 0     + 0
% ...

%% Task 2-a Generating Sampled Signal
close all;
clear variables;

% 8kHz
fs = 8e3;
ts = 1/fs;
%sample for a second
t = 0:ts:1;

%generate the sum of three sinusoids
x = 5*cos(2*pi*500*t)+5*cos(2*pi*1200*t)+5*cos(2*pi*1800*t+0.5*pi);

%plot
plot(x(1:100));

%% Task 2-b Impulse Responses
close all;

h1 = [ -0.0012  -0.0025  -0.0045  -0.0068  -0.0073 ...
       -0.0030   0.0089   0.0297   0.0583   0.0907 ...
        0.1208   0.1422   0.1500   0.1422   0.1208 ...
        0.0907   0.0583   0.0297   0.0089  -0.0030 ...
       -0.0073  -0.0068  -0.0045  -0.0025  -0.0012];
   
h2 =[   0.0004  -0.0017  -0.0064  -0.0076   0.0073 ...
        0.0363   0.0458   0.0000  -0.0802  -0.1134 ...
       -0.0419   0.0860   0.1500   0.0860  -0.0419 ...
       -0.1134  -0.0802   0.0000   0.0458   0.0363 ...
        0.0073  -0.0076  -0.0064  -0.0017   0.0004 ];
    
subplot(2,1,1);
stem(h1);
title("h1 impulse response");
subplot(2,1,2);
stem(h2);
title("h2 impulse response");

%% Task 2-c Filtering and fourier transforms
close all;

%filter input with both sets of coefficients
y1 = myFIR(x, h1);
y2 = myFIR(x, h2);

%fourier transform for original and filtered signal
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

% input has three components, at 500, 1200 and 1800Hz
% h1 is acting as a high pass filter, and removing all components but 500Hz
% h2 is acting as a band pass, removing all components except 1200Hz
%% Task 2-d Playing original signal and h1 filter
sound(x);
pause(2);
sound(y1);

%% Task 2-e Playing original signal and h2 filter
sound(x);
pause(2);
sound(y2);

% this signal is audibly a higher frequency than the unfiltered signal, and
% the signal after filtering with h1, as expected from the frequency
% spectrum plots.

%% Task 3 frequency response using freqz()
close all;
clear variables;

fs =8e3;

% filter coefficients
h1 = [ -0.0012  -0.0025  -0.0045  -0.0068  -0.0073 ...
       -0.0030   0.0089   0.0297   0.0583   0.0907 ...
        0.1208   0.1422   0.1500   0.1422   0.1208 ...
        0.0907   0.0583   0.0297   0.0089  -0.0030 ...
       -0.0073  -0.0068  -0.0045  -0.0025  -0.0012];
h2 =[   0.0004  -0.0017  -0.0064  -0.0076   0.0073 ...
        0.0363   0.0458   0.0000  -0.0802  -0.1134 ...
       -0.0419   0.0860   0.1500   0.0860  -0.0419 ...
       -0.1134  -0.0802   0.0000   0.0458   0.0363 ...
        0.0073  -0.0076  -0.0064  -0.0017   0.0004 ];

% take frequency response for each filter using freqz
[h1, w1] = freqz(h1, 1, 512, fs);
[h2, w2] = freqz(h2, 1, 512, fs);


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

% these results match what was seen in the previous question. h1 is a low
% pass filter and h2 is a band-pass filter
%% Task 4 we.dat
close all;
clear variables;

fs = 8000;

%filter 1
h1 = [ -0.0012  -0.0025  -0.0045  -0.0068  -0.0073 ...
       -0.0030   0.0089   0.0297   0.0583   0.0907 ...
        0.1208   0.1422   0.1500   0.1422   0.1208 ...
        0.0907   0.0583   0.0297   0.0089  -0.0030 ...
       -0.0073  -0.0068  -0.0045  -0.0025  -0.0012];
% filter 2   
h2 =[   0.0004  -0.0017  -0.0064  -0.0076   0.0073 ...
        0.0363   0.0458   0.0000  -0.0802  -0.1134 ...
       -0.0419   0.0860   0.1500   0.0860  -0.0419 ...
       -0.1134  -0.0802   0.0000   0.0458   0.0363 ...
        0.0073  -0.0076  -0.0064  -0.0017   0.0004 ];

% import we.dat file    
x = importdata('WE.DAT');

% filter x with b1 and b2
y1 = myFIR(x, h1);
y1_ = filter(h1, 1, x);
y2 = myFIR(x, h2);
y2_ = filter(h2, 1, x);

% calculate frequency spectrum for original and filtered signals
x_fft = abs(fft(x));
y1_fft = abs(fft(y1));
y2_fft = abs(fft(y2));

% x axis for spectrum plots
f = 0:4:fs/2;

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

% filters are acting as expected, h1 is a low pass so is removing
% frequencies above a certain point. h2 is a band-pass and is rejecting
% high and low frequency, as can be seen by the removal of low frequency
% components.

%% Task 4-b Playing filtered and unfiltered signal
sound(x, fs);
pause(2);
sound(y1, fs);
pause(2);
sound(y2, fs);

% if there is speech in the signal filtering has not helped remove noise
% ebough to make recognisable speech

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

fvtool(lp, hp);
legend('lp', 'hp');
            
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
% Filter 1 - 60Hz
h1 = myFirNotch(fs, f1);
% Filter 2 - 120Hz
h2 = myFirNotch(fs, f2);
% Filter 3 - 180Hz
h3 = myFirNotch(fs, f3);

% two options for combining filters, cascading the filters will apply the
% output of one filter to the input of another 'combining' the filters.
% another option is convolution, which will result in a single set of
% filter coefficients, which is the faster solution processing wise.
h4 = conv(conv(h1,h2),h3);

%Plot frequency response of each filter
fvtool(h1,1,h2,1,h3,1,h4,1, 'Fs', fs);
legend("f1-60Hz","f2-120Hz", "f3-180Hz", "f4-convoluted",'Location', 'SouthEast');

%Convoluted filter is clearly a combination of the individual parts, with a
%notch at each centre frequency for the original 3 filters. It also has
%poles and zeros in the same locations as the originals.

%% Task 6-b Comnining filters and comparing signal spectrums
close all;

%Combine filters by cascading the 60Hz, 120Hz, and 180Hz notch filters
ecg_casc = myFIR(ecg, h1);  %60
ecg_casc = myFIR(ecg_casc, h2); %120
ecg_casc = myFIR(ecg_casc, h3); %180

% output from convoluted filter coefficients
ecg_conv = myFIR(ecg, h4);

% fourier transform of signal before and after filtering
fft_noise = abs(fft(ecg));
fft_casc  = abs(fft(ecg_casc));
fft_conv  = abs(fft(ecg_conv));

%for plotting frequency domain data
fft_half  = length(fft_noise)/2;
df = fs/N;
f = 0:df:fs/2-df;

%plot signal pre and post filtering in time domain
figure(1);
subplot(3,1,1);
plot(ecg, 'b');
title("Noisy ecg signal");
subplot(3,1,2);
plot(ecg_casc, 'r');
title("Cascaded filters");
subplot(3,1,3);
plot(ecg_conv, 'm');
title("Convoluted filters");

%plot frequency spectrum of signal before and after filtering
figure(2);
%unfiltered signal
subplot(3,1,1);
plot(f, fft_noise(1:fft_half), 'b');
title("Spectrum before filtering");
xlabel("Frequency (Hz)");
set(gca, 'YAxisLocation', 'origin');
% cascaded filters
subplot(3,1,2);
plot(f, fft_casc(1:fft_half), 'r');
title("Spectrum of cascaded filters");
set(gca, 'YAxisLocation', 'origin');
xlabel("Frequency (Hz)");
% convoluted filter
subplot(3,1,3);
plot(f, fft_conv(1:fft_half),'m');
title("Spectrum of convoluted filters");
set(gca, 'YAxisLocation', 'origin');
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
% increases the pass band narrows and stop band attenuation increases.

%% Task 8 - IIR Filter
close all;
clear variables;

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

% theta is directly proportional to the location of the centre frequency of
% the pass band, as it increases the location of the passband is higher
%% Task 9-a Analysing pcm.mat
close all;
clear variables;

%import pcm data
pcm = importdata('pcm.mat');

% normalised frequency bin index from -pi to pi
f = (-1:(2/15999):1);
pcm_fft = abs(fftshift(fft(pcm)));

%plot signal in time and frequency domain
subplot(2,1,1);
plot(pcm)
subplot(2,1,2);
plot(f, pcm_fft);
xticks([-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]);
xlim([-1 1]);
xlabel('Normalised \pi rads/sample');
set(gca, 'YAxisLocation', 'origin');
sound(pcm);

% From the spectrum plot, the narrowband frequency is centred around 0.7866pi rads
% The bandwidth is around 0.008pi rads

%% Task 9-b Calculating filter coefficients
close all;

theta = 0.7866*pi;
r = 0.98;

%H(z) = (1-r)/(1-2r*cos(theta)z^-1+r^2*Z^-2)
%H(z) = Y(z)/X(z)
%Y(z)((1-2r*cos(theta)z^-1+r^2*Z^-2) = X(z)(1-r)
%Y(z)-Y(z)1.46z^-1+Y(z)0.81z^-2 = X(z)0.1
% Inverse ZT
%y(n)-y(n-1)1.46+y(n-2)0.8 = x(n)0.1
%y(n)a(1)-y(n-1)a(2)+y(n-2)a(3) = x(n)b(1)

%calculate coefficients
b    = 1-r;
a(1) = 1;
a(2) = -2*r*cos(theta);
a(3) = r^2;

%plot frequency response and pole positions
fvtool(b, a);
%% Task 9-c Filtering pcm.dat
close all;

%filter signal and take fft
pcm_filt = filter(b,a, pcm);
pcm_filt_fft = abs(fftshift(fft(pcm_filt)));

%unfilter time domain
subplot(4,1,1);
plot(pcm);
%filtered time domain
subplot(4,1,3);
plot(pcm_filt);

%unfilteref frequency domain
subplot(4,1,2);
plot(f, pcm_fft);
xticks([-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]);
xlim([-1 1]);
xlabel('Normalised \pi rads/sample');
set(gca, 'YAxisLocation', 'origin');

%filtered frequency domain
subplot(4,1,4);
plot(f, pcm_filt_fft);
xticks([-1 -0.8 -0.6 -0.4 -0.2 0 0.2 0.4 0.6 0.8 1]);
xlim([-1 1]);
xlabel('Normalised \pi rads/sample');
set(gca, 'YAxisLocation', 'origin');

%from the plots the filter signal has significanly less wide band noise in
%the frequency spectrum, and when played the sound is much clearer

sound(pcm_filt);

%% Task 9-d FIR filter equivelants
close all;

% step response for 50, 100 and 150 points
fir1 = zeros(1,50);
fir2 = zeros(1,100);
fir3 = zeros(1,150);
fir1(1) = 1;
fir2(1) = 1;
fir3(1) = 1;

%calculate impulse response
h1 = filter(b,a, fir1);
h2 = filter(b,a, fir2);
h3 = filter(b,a, fir3);

%plot against iir filter
fvtool(b, a, h1, 1, h2, 1, h3, 1);
legend('IIR','FIR1', 'FIR2', 'FIR3');

% The FIR filter responses have far more oscillation in there frequency
% response, even with 150 coefficients. Compared to the smooth respone for
% an IIR with 3 coefficients the FIR filters will take a lot longer to
% process for noisier results.

%% Task 10 IIR plucked string synthesis
clear variables;
close all;

%sample rate and period
fs=8e3;
ts=1/fs;

%iniliase array of length L, with N random numbers at the start
L = 2/ts;
N= 100;
x = [randn(1,N) zeros(1,L)];

%Gain value
K = 0.98;

%set b coefficient to 1
b =1;
%initialise a coefficents as 1
a = ones(1,N);
% set N and N-1 to K/2
a(N) = K/2;
a(N-1) = K/2;

%filter signal
y = filter(b,a,x);

%plot frequency response
freqz(b,a);

%% Task 10-b Playing 'filtered' signal
soundsc(y, fs)