function [ t,q ] = relative_trans_quat_from_first_to_second( t1,q1,t2,q2 )
%RELATIVE_TRANS_QUAT_BETWEEN Summary of this function goes here
%   Detailed explanation goes here
    t = t2-t1;
    q = dcm2quat(quat2dcm(q2)*quat2dcm(q1)');
end

