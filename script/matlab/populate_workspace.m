% t represents translation from origin in inertial frame
% q represents orientation in inertial frame
clear all
folder='../../dataset/rgbd_dataset_freiburg3_long_office_household/';
[timestamp, t(:,1), t(:,2), t(:,3), q(:,1), q(:,2), q(:,3), q(:,4)] = ...
    extract_pos_from_algo([folder 'groundtruth.txt']);

[timestamp_rgb ] = extract_timestamp([folder 'rgb.txt']);
[timestamp_depth ] = extract_timestamp([folder 'depth.txt']);

[timestamp_essai, relativ_trans_essai(:,1), relativ_trans_essai(:,2), relativ_trans_essai(:,3), ...
    relativ_quat_essai(:,1), relativ_quat_essai(:,2), relativ_quat_essai(:,3), relativ_quat_essai(:,4)] ...
    = extract_pos_from_algo('../../bin/relativ_pose.txt');
    
[timestamp_literature, relativ_trans_literature(:,1), relativ_trans_literature(:,2), relativ_trans_literature(:,3), ...
    relativ_quat_literature(:,1), relativ_quat_literature(:,2), relativ_quat_literature(:,3), relativ_quat_literature(:,4)] ...
    = extract_pos_from_algo('../../dataset/freiburg1_xyz-rgbdslam.txt');
relativ_trans_literature = relativ_trans_literature - ones(length(relativ_trans_literature),1)*relativ_trans_literature(1,:); 

[timestamp_essai, t_essai(:,1), t_essai(:,2), t_essai(:,3), ...
    q_essai(:,1), q_essai(:,2), q_essai(:,3), q_essai(:,4)] ...
    = extract_pos_from_algo('../../bin/pose.txt');

dt_ground_truth = diff(timestamp);
dt_essai = diff(timestamp_essai);

t0=timestamp(1);
timestamp = timestamp - ones(length(timestamp),1)*t0;
timestamp_essai = timestamp_essai - ones(length(timestamp_essai),1) * t0;
timestamp_literature = timestamp_literature - ones(length(timestamp_literature),1) * t0;

for i=2:length(t)
    [relativ_trans(i-1,:), relativ_quat(i-1,:)] = relative_trans_quat_from_first_to_second(t(i-1,:),q(i-1,:),t(i,:),q(i,:));
end

%% Vitesse
v(:,1) = relativ_trans(:,1)./dt_ground_truth;
v(:,2) = relativ_trans(:,2)./dt_ground_truth;
v(:,3) = relativ_trans(:,3)./dt_ground_truth;
norm_v = sqrt(v(:,1).^2+v(:,2).^2+v(:,3).^2);

v_essai(:,1) = relativ_trans_essai(2:end,1)./dt_essai;
v_essai(:,2) = relativ_trans_essai(2:end,2)./dt_essai;
v_essai(:,3) = relativ_trans_essai(2:end,3)./dt_essai;
norm_v_essai = sqrt(v_essai(:,1).^2+v_essai(:,2).^2+v_essai(:,3).^2);

%%
norm_v_essai(norm_v_essai==0) = NaN;
windowSize = 1;
norm_v_essai_lisse = filter(ones(1,windowSize)/windowSize,1,norm_v_essai);

plot(timestamp_essai(2:end),norm_v_essai_lisse,'b+');
hold on;
plot(timestamp(2:end),norm_v,'r')

%%
figure
IndexInit = find(abs(timestamp - timestamp_essai(1)) < 0.020 ,1,'first');
t_ground_init = t(IndexInit,:);
q_ground_init = q(IndexInit,:);

subplot(3,1,1);
hold on;
plot(timestamp,t(:,1)-ones(length(t),1) * t_ground_init(:,1),'r');
plot(timestamp_essai,t_essai(:,1));

subplot(3,1,2);
hold on;    
plot(timestamp,t(:,2)-ones(length(t),1) * t_ground_init(:,2),'r');
plot(timestamp_essai,t_essai(:,2));

subplot(3,1,3);
hold on;
plot(timestamp,t(:,3)-ones(length(t),1)*t_ground_init(:,3),'r');
plot(timestamp_essai,t_essai(:,3));

%% Quat
figure
plot(timestamp_essai,q_essai);
hold on
plot(timestamp,q);