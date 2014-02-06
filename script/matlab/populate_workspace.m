% t represents translation from origin in inertial frame
% q represents orientation in inertial frame
clear all
[timestamp, t(:,1), t(:,2), t(:,3), q(:,1), q(:,2), q(:,3), q(:,4)] = extract_log_ground_truth_format('../../dataset/groundtruth.txt');
[timestamp_rgb ] = extract_timestamp('../../dataset/rgb.txt');
[timestamp_depth ] = extract_timestamp('../../dataset/depth.txt');

[timestamp_essai, relativ_trans_essai(:,1), relativ_trans_essai(:,2), relativ_trans_essai(:,3), ...
    q_essai(:,1), q_essai(:,2), q_essai(:,3), q_essai(:,4)] ...
    = extract_pos_from_algo('../../bin/pose.txt');

dt_ground_truth = diff(timestamp);
dt_essai = diff(timestamp_essai);

for i=2:length(t)
    [relativ_trans(i-1,:), relativ_quat(i-1,:)] = relative_trans_quat_from_first_to_second(t(i-1,:),q(i-1,:),t(i,:),q(i,:));
end

% Vitesse
v(:,1) = relativ_trans(:,1)./diff(timestamp);
v(:,2) = relativ_trans(:,2)./diff(timestamp);
v(:,3) = relativ_trans(:,3)./diff(timestamp);
norm_v = sqrt(v(:,1).^2+v(:,2).^2+v(:,3).^2);

v_essai(:,1) = relativ_trans_essai(2:end,1)./diff(timestamp_essai);
v_essai(:,2) = relativ_trans_essai(2:end,2)./diff(timestamp_essai);
v_essai(:,3) = relativ_trans_essai(2:end,3)./diff(timestamp_essai);
norm_v_essai = sqrt(v_essai(:,1).^2+v_essai(:,2).^2+v_essai(:,3).^2);
%%
norm_v_essai(norm_v_essai==0) = NaN;
windowSize = 10;
norm_v_essai_lisse = filter(ones(1,windowSize)/windowSize,1,norm_v_essai);

plot(timestamp_essai(2:end),norm_v_essai_lisse,'b');
hold on;
plot(timestamp(2:end),norm_v,'r')