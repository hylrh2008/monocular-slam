function [qout] = dcm2quat(DCM)
%Compute the euler parameters from the direction cosine matrix
% [qout] = dcm2quat(DCM)
%
% Author: Karl Ludwig Fetzer

qout = zeros(4,1);

% Scalar term
qout(4) = 0.5 * sqrt(1 + DCM(1,1) + DCM(2,2) + DCM(3,3));

% vector term
qout(1) = (DCM(3,2) - DCM(2,3)) / ( 4 * qout(4) );
qout(2) = (DCM(1,3) - DCM(3,1)) / ( 4 * qout(4) );
qout(3) = (DCM(2,1) - DCM(1,2)) / ( 4 * qout(4) );

end