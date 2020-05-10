function [b] = myFirNotch(fs,fo)
%myFirNotch calculate filter coefficents for 2nd order FIR notch filter
%   Uses zero and pole placement to calculate a set of filter coefficients
%   for a 2nd order FIR notch filter
%   fs - sampling frequency
%   fo - notch centre frequency
%   b  - filter coefficients

%initiliase array
b = zeros(1,3);

%calculate freqeuncies position on unit circle in radians
wo = (2*pi*fo)/fs;

%This gives a pair of complex conjugate zeros
%H'(z) = (z-e^jwo)(z-e^0jwo)

%Expanding the brackets and appying euler gives:
%z^2-2cos(wo)z+1

%add two poles at z=0 as the degree of the numerator must be >= degree of
%denominator
%H(z) = H'(z)/z^2

%Dividing numerator by z^2 removes all positive z powers, giving:
%H(z) = 1-2cos(wo)z^-1+z^-2

%inverse z transform z becomes x(n-power of z)
%h(n) = x(n)-2cos(wo)x(n-1)+x(n-2)
%h(n) = b(0)x(n)+b(1)x(n-1)+b(2)x(n-2)

%From this the filter coefficient b can be extrapolated as:
b(1) = 1;           %x(n)   
b(2) = -2*cos(wo);  %x(n-1)
b(3) = 1;           %x(n-2)

end

