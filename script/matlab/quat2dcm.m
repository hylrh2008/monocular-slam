function [DCM] = quat2dcm(q_in)
%Compute the direction cosine matrix from Euler Parameters
% [DCM] = quat2dcm(q_in)
%
% Author: Karl Ludwig Fetzer

DCM(1,1) = 1 - 2 * q_in(2)^2 - 2 * q_in(3)^2;
DCM(1,2) = 2 * (q_in(1)*q_in(2) - q_in(3)*q_in(4));
DCM(1,3) = 2 * (q_in(3)*q_in(1) + q_in(2)*q_in(4));
DCM(2,1) = 2 * (q_in(1)*q_in(2) + q_in(3)*q_in(4));
DCM(2,2) = 1 - 2 * q_in(3)^2 - 2 * q_in(1)^2;
DCM(2,3) = 2 * (q_in(2)*q_in(3) - q_in(1)*q_in(4));
DCM(3,1) = 2 * (q_in(3)*q_in(1) - q_in(2)*q_in(4));
DCM(3,2) = 2 * (q_in(2)*q_in(3) + q_in(1)*q_in(4));
DCM(3,3) = 1 - 2 * q_in(1)^2 - 2 * q_in(2)^2;

end
