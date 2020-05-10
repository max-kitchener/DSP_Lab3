function [y] = myFIR(x,h)
%  MYFIR FIR filter of signal x, with impulse response h
% 
% x - Input signal
% h - Impulse response
% y - FIltered output


%calculate the length of each array
M = length(h)-1;
N = length(x)-1;
y = zeros(1, length(x));

% for each sample n
for n = 0:N
    % clear sum
    sum = 0;
    % for each filter coefficient
    for k = 0:M
        % reset calc
        calc = 0;
        % only process if array indexes are valid
        if n-k >= 0 
            % h(k)*x(n-k)
            calc = h(k+1)*x((n-k)+1);
        end
        % sum values
        sum = sum + calc;
    end
    % write filtered value to output array
    y(n+1) = sum;
end
end % end of function