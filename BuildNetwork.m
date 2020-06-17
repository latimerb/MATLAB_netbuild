close all;
clear all;
clc;

% This script will generate connectivity based on pre defined cell
% positions and connectivity rules.

% Set random seed
%%seed = 123421;
%rng(seed);
%stream = RandStream('mt19937ar', 'Seed', seed);
%
%% Set number of workers
%parpool(24);
%
%%%% TODO:
%%%% 1) Reciprocal connectivity
%%%% 2) Assign weights
%
%%%%%%%%%%%%%%%%%%%%%%%%%% Network parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%num_cells = 72000; 
%num_pyrA = 28801; 
%num_pyrC = 28801; 
%num_axo = 1420; 
%num_pv = 12978;

%% geometry
%length = 1400;
%width = 1400;
%height = 1400;
%
%% Load or generate positions
%%dirname = 'positions.csv';
%%pos = importdata(dirname);
%%pos = [pos(1:num_cells,:),[0:1:num_cells-1]'];
%
%pos = generate_positions(num_pyrA+num_pyrC,num_axo,num_pv,...
%                        width,length,height,seed);
%
%sprintf('pos size: %d',size(pos,1))
%
%% Conduction velocity
%cond_vel = 500; %um/ms
%
%% Connectivity
%PN2FSI = 0.12;
%FSI2PN = 0.34;
%AXO2FSI = 0;
%PN2PN = 0;
%FSI2FSI = 0.26;
%PN2AXOrecip_perc = 0.4; %reciprocal
%
%% Weights (mean)
%PN2AXOmean = 0.00235; PN2AXOstd = 0.1*PN2AXOmean;
%PN2PVmean = 0.002; PN2PVstd = 0.1*PN2PVmean;
%PN2PNmean = 0; PN2PNstd = 0.1*PN2PNmean;
%
%PV2PVmean = 0.004; PV2PVstd = 0.1*PV2PVmean;
%PV2AXOmean = 0.008; PV2AXOstd = 0.1*PV2AXOmean;
%PV2PNmean = 0.008; PV2PNstd = 0.1*PV2PNmean;
%
%AXO2PVmean = 0.0; AXO2PVstd = 0.1*AXO2PVmean;
%AXO2AXOmean = 0.0; AXO2AXOstd = 0.1*AXO2AXOmean;
%AXO2PNmean = 0.003; AXO2PNstd = 0.2*AXO2PNmean; 
%
%
%%%%%%%%%%%%%%%%%%%%%%%% Initialize matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% matrices for outputs (connections, weights, and delays)
%connout = cell(num_cells,1);
%weightout = cell(num_cells,1);
%delayout = cell(num_cells,1);
%
%PN2AXOrecip_conns = NaN(num_pyrA+num_pyrC,1000);
%
%% indices of cells
%pyrA = int32(floor([0,num_pyrA-1]));
%pyrC = int32(floor([num_pyrA,num_pyrA+num_pyrC-1]));
%axo = int32(floor([num_pyrA+num_pyrC, num_pyrA+num_pyrC+num_axo-1]));
%bask = int32(floor([num_pyrA+num_pyrC+num_axo, num_cells-1]));



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Generate connectivity based on percentages in literature and store
%%%% them in CONNOUT. This is a pre-centric matrix (the rows are 
%%%% presynaptic cells). However, the network model needs the rows to be 
%%%% POSTsynaptic cells. This will be done in step 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate positions
%pos = generate_positions(num_pyrA+num_pyrC,num_axo,num_pv,length,width,height,seed);


%parfor i=0:num_cells-1
%    
%    t = getCurrentTask();
%    if t.ID == 1
%        sprintf('Building connections. %d percent done',100*i/(num_cells-1))
%    end
%    
%    radius = 300; %um
%    cell_pos = pos(i+1,2:4); %projecting cell's position
%    
%    x2 = pos(:,2); y2 = pos(:,3); z2 = pos(:,4);
%    x1 = cell_pos(1); y1 = cell_pos(2); z1 = cell_pos(3);
%    
%    % possible connections
%    possible_conns_ID = pos(sqrt((x2-x1).^2+(y2-y1).^2+(z2-z1).^2)<=radius,1);
%    
%    % cannot connect to itself
%    possible_conns_ID(possible_conns_ID==i)=[];
%    
%    
%    % First, get all possible connections based on distance (above)
%    % Then break those IDs into types
%    possible_INT_ID = possible_conns_ID(possible_conns_ID>=axo(1) & possible_conns_ID<=num_cells-1);
%    possible_PN_ID = possible_conns_ID(possible_conns_ID>=pyrA(1) & possible_conns_ID<=pyrC(2));
%    
%%     scatter3(pos(possible_INT_ID+1,1),pos(possible_INT_ID+1,2),pos(possible_INT_ID+1,3),'g.');hold on;
%%     scatter3(pos(possible_PN_ID+1,1),pos(possible_PN_ID+1,2),pos(possible_PN_ID+1,3),'r.');hold on;
%%     scatter3(cell_pos(1),cell_pos(2),cell_pos(3),'m*')
%%     xlim([0 1400]);ylim([0 1400]);zlim([0 1400]);
%    
%    % If presynaptic cell is pyramidal, do the following
%    if i>=pyrA(1) && i<=pyrC(2) 
%        % Find connections
%        num_PN = ceil(PN2PN*size(possible_PN_ID,1));
%        num_INTS = ceil(PN2FSI*size(possible_INT_ID,1));
%        outgoingPNconns = sortrows(datasample(stream,possible_PN_ID,num_PN,'Replace',false));
%        outgoingINTconns = sortrows(datasample(stream,possible_INT_ID,num_INTS,'Replace',false));
%        outgoingAXOconns = outgoingINTconns(outgoingINTconns>=axo(1) & outgoingINTconns<=axo(2));
%        outgoingPVconns = outgoingINTconns(outgoingINTconns>=bask(1) & outgoingINTconns<=bask(2));
%        
%        outgoingconns = cat(1,outgoingPNconns,outgoingINTconns); %concatenate
%        
%        connout{i+1} = outgoingvec(outgoingconns,i);
%        
%        % Some need to be reciprocal, store those here
%%         PN2AXOrecip_num = ceil(PN2AXOrecip_perc*size(outgoingAXOconns,1));
%%         if PN2AXOrecip_num>0
%%             PN2AXOrecip_conns(i+1,1:PN2AXOrecip_num) = sortrows(datasample(outgoingAXOconns,PN2AXOrecip_num,'Replace',false))';
%%         end
%        
%        % Now get axonal delay
%        pos_outgoingconns = pos(ismember(pos(:,4),outgoingconns),:);
%        dist = (pos_outgoingconns(:,1:3)-repmat(cell_pos,size(pos_outgoingconns,1),1)).^2;
%        dist = sqrt(sum(dist,2)); % distance in microns
%        delays = dist/cond_vel; % divide by conduction velocity
%        delays(delays<0.1) = 0.1; % set minimum delay (should be > dt in NEURON)
%        
%        delayout{i+1} = outgoingvec(delays,i);
%        
%        % Finally, assign weights
%        weights = zeros(size(outgoingconns,1),1);
%        % Transform mean and std for lognormal
%        mu = log((PN2AXOmean^2)/sqrt(PN2AXOstd^2+PN2AXOmean^2));
%        sigma = sqrt(log(PN2AXOstd^2/(PN2AXOmean^2)+1));
%        weights(outgoingconns>=axo(1) & outgoingconns<=axo(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingAXOconns,1)])',6);
%        % Transform mean and std for lognormal
%        mu = log((PN2PVmean^2)/sqrt(PN2PVstd^2+PN2PVmean^2));
%        sigma = sqrt(log(PN2PVstd^2/(PN2PVmean^2)+1));
%        weights(outgoingconns>=bask(1) & outgoingconns<=bask(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingPVconns,1)])',6);
%        
%        weightout{i+1} = outgoingvec(weights,i);
%    end
%    
%    % If presynaptic cell is axo, do this
%    if i>=axo(1) && i<=axo(2) 
%        %Find connections
%        num_PN = ceil(FSI2PN*size(possible_PN_ID,1));
%        num_INTS =  ceil(AXO2FSI*size(possible_INT_ID,1));
%        outgoingPNconns = sortrows(datasample(stream,possible_PN_ID,num_PN,'Replace',false));
%        outgoingINTconns = sortrows(datasample(stream,possible_INT_ID,num_INTS,'Replace',false));
%        outgoingAXOconns = outgoingINTconns(outgoingINTconns>=axo(1) & outgoingINTconns<=axo(2));
%        outgoingPVconns = outgoingINTconns(outgoingINTconns>=bask(1) & outgoingINTconns<=bask(2));
%        
%        outgoingconns = cat(1,outgoingPNconns,outgoingINTconns); %concatenate
%        connout{i+1} = outgoingvec(outgoingconns,i);
%        
%        % Now get axonal delay
%        pos_outgoingconns = pos(ismember(pos(:,4),outgoingconns),:);
%        dist = (pos_outgoingconns(:,1:3)-repmat(cell_pos,size(pos_outgoingconns,1),1)).^2;
%        dist = sqrt(sum(dist,2)); % distance in microns
%        delays = dist/cond_vel; % divide by conduction velocity
%        delays(delays<0.1) = 0.1; % set minimum delay (should be > dt in NEURON)
%        
%        delayout{i+1} = outgoingvec(delays,i);
%        
%        % Finally, assign weights
%        weights = nan(size(outgoingconns,1),1);
%        
%        % Transform mean and std for lognormal
%        [mu, sigma] = logtrans(AXO2AXOmean,AXO2AXOstd);
%        weights(outgoingconns>=axo(1) & outgoingconns<=axo(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingAXOconns,1)])',6);
%        [mu, sigma] = logtrans(AXO2PVmean,AXO2PVstd);
%        weights(outgoingconns>=bask(1) & outgoingconns<=bask(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingPVconns,1)])',6);
%        [mu, sigma] = logtrans(AXO2PNmean,AXO2PNstd);
%        weights(outgoingconns>=pyrA(1) & outgoingconns<=pyrC(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingPNconns,1)])',6);
%        
%        weightout{i+1} = outgoingvec(weights,i);
%    end
%    
%    if i>=bask(1) && i<=bask(2) 
%        %Find connections
%        num_PN = ceil(FSI2PN*size(possible_PN_ID,1));
%        num_INTS =  ceil(FSI2FSI*size(possible_INT_ID,1));
%        outgoingPNconns = sortrows(datasample(stream,possible_PN_ID,num_PN,'Replace',false));
%        outgoingINTconns = sortrows(datasample(stream,possible_INT_ID,num_INTS,'Replace',false));
%        outgoingAXOconns = outgoingINTconns(outgoingINTconns>=axo(1) & outgoingINTconns<=axo(2));
%        outgoingPVconns = outgoingINTconns(outgoingINTconns>=bask(1) & outgoingINTconns<=bask(2));
%        
%        outgoingconns = cat(1,outgoingPNconns,outgoingINTconns); %concatenate
%        connout{i+1} = outgoingvec(outgoingconns,i);
%        
%        % Now get axonal delay
%        pos_outgoingconns = pos(ismember(pos(:,4),outgoingconns),:);
%        dist = (pos_outgoingconns(:,1:3)-repmat(cell_pos,size(pos_outgoingconns,1),1)).^2;
%        dist = sqrt(sum(dist,2)); % distance in microns
%        delays = dist/cond_vel; % divide by conduction velocity
%        delays(delays<0.1) = 0.1; % set minimum delay (should be > dt in NEURON)
%        
%        delayout{i+1} = outgoingvec(delays,i);
%        
%        % Finally, assign weights
%        weights = zeros(size(outgoingconns,1),1);
%        
%        [mu,sigma] = logtrans(PV2AXOmean,PV2AXOstd);
%        weights(outgoingconns>=axo(1) & outgoingconns<=axo(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingAXOconns,1)])',6);
%        [mu, sigma] = logtrans(PV2PVmean,PV2PVstd);
%        weights(outgoingconns>=bask(1) & outgoingconns<=bask(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingPVconns,1)])',6);
%        [mu, sigma] = logtrans(PV2PNmean,PV2PNstd);
%        weights(outgoingconns>=pyrA(1) & outgoingconns<=pyrC(2),1) = ...
%            round(random('logn',mu,sigma,[1,size(outgoingPNconns,1)])',6);
%        
%        weightout{i+1} = outgoingvec(weights,i);
%    end
%    
%end
%
%% create adjacency matrix and visualize connections of particular cell for
%% verification
%% init = 9968; % presynaptic cell of interest (gid)
%% adj_effpyr = zeros(num_cells,num_cells);
%% for i=3:3+connout(init+1,1)-1
%%     k = connout(init+1,i)+1; % gid needs 1 added
%%     adj_effpyr(init+1,k) = 1;
%% end
%% 
%% gplot(adj_effpyr,pos(:,2:4),'b');
%% xlim([0 1400]);ylim([0 1400]);zlim([0 1400]);
%
%% Position output file
%
%fileID = fopen('position.dat','w');
%fprintf(fileID,'%d\n',num_cells); %first line should be number of cells
%
%for i=0:num_cells-1
%   fprintf(fileID,'%d\t%d\t%d',pos(i+1,2),pos(i+1,3),pos(i+1,4));
%   fprintf(fileID,'\n');
%end
%
%fclose(fileID);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Take the pre-centric matrix and convert it to a post-centric one.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sprintf('flipping matrix')
%
%clearvars -except connout weightout num_cells
%
%connout = cell2mat(connout);
%weightout = cell2mat(weightout);
%
%save('connout.mat', 'connout', '-v7.3')
%save('weightout.mat', 'weightout', '-v7.3')

load('connout.mat')
load('weightout.mat')

%afferentconn = cell(num_cells,1);
%afferentwgt = cell(num_cells,1);
%for i=20000:25000
%    tic
%    % Get all connections to postsynaptic cell (skip first two columns)
%    afferentconn=cell(1,1);
%    afferentwgt=cell(1,1);
% 
%    afferentconn{1}=flipmatconn(connout,i);
%    afferentwgt{1}=flipmatwgt(connout,weightout,i);
%    
%    save(sprintf('./data/afferentconn%d',i),'afferentconn')
%    save(sprintf('./data/afferentwgt%d',i),'afferentwgt')
%    clearvars afferentconn afferentwgt
%    toc
    %afferentconn_inds = find(connout(:,3:end)==i);
    
    % These indices are numbers from
    % 1 to (size(connout,1)*size(connout,2)), numbered down the columns
    %a = connout(:,3:end);
    %afferentconn(i+1,1) = size(afferentconn_inds,1);
    %afferentconn(i+1,2) = i;
    %afferentconn(i+1,3:size(afferentconn_inds,1)+2) = ...
    %    mod(a(afferentconn_inds),size(connout,1))';
    
    %w = weightout(:,3:end);
    %afferentwgt(i+1,1) = size(afferentconn_inds,1);
    %afferentwgt(i+1,2) = i; 
    %afferentwgt(i+1,3:size(afferentconn_inds,1)+2) = ...
    %    mod(w(afferentconn_inds),size(connout,1))';
%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Write the output files.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sprintf('writing output files')
%
%afferentconn = cell2mat(afferentconn);
%afferentwgt = cell2mat(afferentwgt);
%
%% Connectivity output file
num_cells = 72000;

num_cores=24;
cells_per_core = num_cells/num_cores;

tID = 4
tic
sprintf('tID: %d',tID)
fileID1 = fopen(sprintf('connectivity%d.dat',tID),'w');
fprintf(fileID1,'%d\n',num_cells); %first line should be number of cells

fileID2 = fopen(sprintf('weights%d.dat',tID),'w');
fprintf(fileID2,'%d\n',num_cells); %first line should be number of cells

sprintf('made fileIDs')   
for i=(tID-1)*cells_per_core:tID*cells_per_core

	sprintf('starting for loop') 
	ac = cell(1,1);
	aw = cell(1,1);

	sprintf('i: %d',i)
	ac{1}=flipmatconn(connout,i);

	fprintf(fileID1,'%d\t',ac{1}(1,~isnan(ac{1}(1,1:end-1)))); 
	fprintf(fileID1,'%d',ac{1}(1,~isnan(ac{1}(1,end))));
	fprintf(fileID1,'\n');

	sprintf('done writing conn')

	aw{1}=flipmatwgt(connout,weightout,i);
	fprintf(fileID2,'%d\t',aw{1}(1,~isnan(aw{1}(1,1:end-1))));
	fprintf(fileID2,'%d',aw{1}(1,~isnan(aw{1}(1,end))));
	fprintf(fileID2,'\n');

	sprintf('done writing wgt')

end
toc
fclose(fileID1);
fclose(fileID2);

%% Weights output file
%fileID = fopen('weights.dat','w');
%fprintf(fileID,'%d\n',num_cells); %first line should be number of cells
%
%for i=0:num_cells-1
%   fprintf(fileID,'%d\t',afferentwgt(i+1,~isnan(afferentwgt(i+1,1:end-1))));
%   fprintf(fileID,'%d',afferentwgt(i+1,~isnan(afferentwgt(i+1,end))));
%   fprintf(fileID,'\n');
%end
%
%fclose(fileID);




% Shut down parallel pool
% delete(gcp('nocreate'))
 
%% Trust but verify (connectivity)

% Pyramidal -> FSI




% Break out positions by cell type
% pyrA_pos = pos(find(pos(:,1)>=pyrA(1) & pos(:,1)<=pyrA(2)),1:4);
% pyrC_pos = pos(find(pos(:,1)>=pyrC(1) & pos(:,1)<=pyrC(2)),1:4);
% chn_pos = pos(find(pos(:,1)>=axo(1) & pos(:,1)<=axo(2)),1:4);
% bask_pos = pos(find(pos(:,1)>=bask(1) & pos(:,1)<=bask(2)),1:4);
% 
% figure
% hold all
% scatter3(pyrA_pos(:,2),pyrA_pos(:,3),pyrA_pos(:,4),'m.');
% scatter3(pyrC_pos(:,2),pyrC_pos(:,3),pyrC_pos(:,4),'r.');
% scatter3(chn_pos(:,2),chn_pos(:,3),chn_pos(:,4),'b.');
% scatter3(bask_pos(:,2),bask_pos(:,3),bask_pos(:,4),'g.');
