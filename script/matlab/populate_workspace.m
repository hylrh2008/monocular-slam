% t represents translation from origin in inertial frame
% q represents orientation in inertial frame
clear all
[timestamp, t(:,1), t(:,2), t(:,3), q(:,1), q(:,2), q(:,3), q(:,4)] = extract_log_ground_truth_format('../../dataset/groundtruth.txt');
[timestamp_rgb ] = extract_timestamp('../../dataset/rgb.txt');
[timestamp_depth ] = extract_timestamp('../../dataset/depth.txt');

dt = diff(timestamp);
for i=2:length(t)
    [relativ_trans(i-1,:), relativ_quat(i-1,:)] = relative_trans_quat_from_first_to_second(t(i-1,:),q(i-1,:),t(i,:),q(i,:));
end

% Vitesse
v(:,1) = relativ_trans(:,1)./diff(timestamp);
v(:,2) = relativ_trans(:,2)./diff(timestamp);
v(:,3) = relativ_trans(:,3)./diff(timestamp);