close all;
clear all;
clc;

% This script will generate connectivity based on pre defined cell
% positions and connectivity rules.

% Set random seed
seed = 123421;
rng(seed);
stream = RandStream('mt19937ar', 'Seed', seed);

% Set number of workers
parpool(4);

%%% TODO:
%%% 1) Reciprocal connectivity
%%% 2) Assign weights

%%%%%%%%%%%%%%%%%%%%%%%%% Network parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_cells = 343000; 
num_pyrA = 137200; 
num_pyrC = 137200; 
num_axo = 6860; 
num_pv = 61740;

% geometry
length = 1400;
width = 1400;
height = 1400;

% Load or generate positions
%dirname = 'positions.csv';
%pos = importdata(dirname);
%pos = [pos(1:num_cells,:),[0:1:num_cells-1]'];

pos = generate_positions(num_pyrA+num_pyrC,num_axo,num_pv,...
                       width,length,height,seed);

sprintf('pos size: %d',size(pos,1))

% Conduction velocity
cond_vel = 500; %um/ms

% Connectivity
PN2FSI = 0.12; %FSIs receive connections from 12% of surrounding PNs
PN2AXO = 0.12;
FSI2PN = 0.34; %PNs receive connections from 34% of surrounding FSIs
AXO2FSI = 0;
PN2PN = 0;
FSI2FSI = 0.26; %FSIs receive connections from 26% of surrounding FSIs
FSI2AXO = 0.26;
PN2AXOrecip_perc = 0.4; %reciprocal
GAPCONN = 0.08;

% Weights (mean)
PN2AXOmean = 0.002; PN2AXOstd = 0.1*PN2AXOmean;
PN2PVmean = 0.002; PN2PVstd = 0.1*PN2PVmean;
PN2PNmean = 0; PN2PNstd = 0.1*PN2PNmean;

PV2PVmean = 0.004; PV2PVstd = 0.1*PV2PVmean;
PV2AXOmean = 0.008; PV2AXOstd = 0.1*PV2AXOmean;
PV2PNmean = 0.008; PV2PNstd = 0.1*PV2PNmean;

AXO2PVmean = 0.0; AXO2PVstd = 0.1*AXO2PVmean;
AXO2AXOmean = 0.0; AXO2AXOstd = 0.1*AXO2AXOmean;
AXO2PNmean = 0.003; AXO2PNstd = 0.2*AXO2PNmean; 


%%%%%%%%%%%%%%%%%%%%%%% Initialize matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% matrices for outputs (connections, weights, and delays)
connin = cell(num_cells,1);
weightin = cell(num_cells,1);
delayin = cell(num_cells,1);
gaps = cell(num_cells,1);

PN2AXOrecip_conns = NaN(num_pyrA+num_pyrC,1000);

% indices of cells
pyrA = int32(floor([0,num_pyrA-1]));
pyrC = int32(floor([num_pyrA,num_pyrA+num_pyrC-1]));
axo = int32(floor([num_pyrA+num_pyrC, num_pyrA+num_pyrC+num_axo-1]));
bask = int32(floor([num_pyrA+num_pyrC+num_axo, num_cells-1]));



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Generate connectivity based on percentages in literature and store
%%%% them in CONNOUT. This is a pre-centric matrix (the rows are 
%%%% presynaptic cells). However, the network model needs the rows to be 
%%%% POSTsynaptic cells. This will be done in step 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Generate positions
pos = generate_positions(num_pyrA+num_pyrC,num_axo,num_pv,length,width,height,seed);


parfor i=0:num_cells-1
   
  
   sprintf('Building connections. %d percent done',100*i/(num_cells-1))

   
   radius = 300; %um
   cell_pos = pos(i+1,2:4); %projecting cell's position
   
   x2 = pos(:,2); y2 = pos(:,3); z2 = pos(:,4);
   x1 = cell_pos(1); y1 = cell_pos(2); z1 = cell_pos(3);
   
   % possible connections
   possible_conns_ID = pos(sqrt((x2-x1).^2+(y2-y1).^2+(z2-z1).^2)<=radius,1);
   
   % cannot connect to itself
   possible_conns_ID(possible_conns_ID==i)=[];
   
   
   % First, get all possible connections based on distance (above)
   % Then break those IDs into types
   possible_INT_ID = possible_conns_ID(possible_conns_ID>=axo(1) & possible_conns_ID<=num_cells-1);
   possible_PN_ID = possible_conns_ID(possible_conns_ID>=pyrA(1) & possible_conns_ID<=pyrC(2));
   possible_BASK_ID = possible_conns_ID(possible_conns_ID>=bask(1) & possible_conns_ID<=bask(2));
   possible_AXO_ID = possible_conns_ID(possible_conns_ID>=axo(1) & possible_conns_ID<=axo(2));
%     scatter3(pos(possible_INT_ID+1,1),pos(possible_INT_ID+1,2),pos(possible_INT_ID+1,3),'g.');hold on;
%     scatter3(pos(possible_PN_ID+1,1),pos(possible_PN_ID+1,2),pos(possible_PN_ID+1,3),'r.');hold on;
%     scatter3(cell_pos(1),cell_pos(2),cell_pos(3),'m*')
%     xlim([0 1400]);ylim([0 1400]);zlim([0 1400]);
   
   % If postsynaptic cell is pyramidal, do the following
   if i>=pyrA(1) && i<=pyrC(2) 
       % Find connections
       num_PN = ceil(PN2PN*size(possible_PN_ID,1));
       num_INTS = ceil(FSI2PN*size(possible_INT_ID,1));
       incomingPNconns = sortrows(datasample(stream,possible_PN_ID,num_PN,'Replace',false));
       incomingINTconns = sortrows(datasample(stream,possible_INT_ID,num_INTS,'Replace',false));
       incomingAXOconns = incomingINTconns(incomingINTconns>=axo(1) & incomingINTconns<=axo(2));
       incomingPVconns = incomingINTconns(incomingINTconns>=bask(1) & incomingINTconns<=bask(2));
       
       incomingconns = cat(1,incomingPNconns,incomingINTconns); %concatenate
       
       connin{i+1} = outgoingvec(incomingconns,i);
       
       % Some need to be reciprocal, store those here
%         PN2AXOrecip_num = ceil(PN2AXOrecip_perc*size(outgoingAXOconns,1));
%         if PN2AXOrecip_num>0
%             PN2AXOrecip_conns(i+1,1:PN2AXOrecip_num) = sortrows(datasample(outgoingAXOconns,PN2AXOrecip_num,'Replace',false))';
%         end
       
       % Now get axonal delay
       pos_incomingconns = pos(ismember(pos(:,1),incomingconns),:);
       dist = (pos_incomingconns(:,2:4)-repmat(cell_pos,size(pos_incomingconns,1),1)).^2;
       dist = sqrt(sum(dist,2)); % distance in microns
       delays = dist/cond_vel; % divide by conduction velocity
       delays(delays<0.1) = 0.1; % set minimum delay (should be > dt in NEURON)
       
       delayin{i+1} = outgoingvec(delays,i);
       
       % Finally, assign weights
       weights = zeros(size(incomingconns,1),1);
       % Transform mean and std for lognormal
       mu = log((AXO2PNmean^2)/sqrt(AXO2PNstd^2+AXO2PNmean^2));
       sigma = sqrt(log(AXO2PNstd^2/(AXO2PNmean^2)+1));
       weights(incomingconns>=axo(1) & incomingconns<=axo(2),1) = ...
           round(random('logn',mu,sigma,[1,size(incomingAXOconns,1)])',6);
       % Transform mean and std for lognormal
       mu = log((PV2PNmean^2)/sqrt(PN2PVstd^2+PN2PVmean^2));
       sigma = sqrt(log(PN2PVstd^2/(PN2PVmean^2)+1));
       weights(incomingconns>=bask(1) & incomingconns<=bask(2),1) = ...
           round(random('logn',mu,sigma,[1,size(incomingPVconns,1)])',6);
       
       weightin{i+1} = outgoingvec(weights,i);
   end
   
   % If postsynaptic cell is axo, do this
   if i>=axo(1) && i<=axo(2) 
       %Find connections
       num_PN = ceil(PN2AXO*size(possible_PN_ID,1));
       num_BASK =  ceil(FSI2AXO*size(possible_BASK_ID,1));
       incomingPNconns = sortrows(datasample(stream,possible_PN_ID,num_PN,'Replace',false));
       incomingBASKconns = sortrows(datasample(stream,possible_BASK_ID,num_BASK,'Replace',false));
       
       incomingconns = cat(1,incomingPNconns,incomingBASKconns); %concatenate
       connin{i+1} = outgoingvec(incomingconns,i);
       
       % Now get axonal delay
       pos_incomingconns = pos(ismember(pos(:,1),incomingconns),:);
       dist = (pos_incomingconns(:,2:4)-repmat(cell_pos,size(pos_incomingconns,1),1)).^2;
       dist = sqrt(sum(dist,2)); % distance in microns
       delays = dist/cond_vel; % divide by conduction velocity
       delays(delays<0.1) = 0.1; % set minimum delay (should be > dt in NEURON)
       
       delayin{i+1} = outgoingvec(delays,i);
       
       % Finally, assign weights
       weights = nan(size(incomingconns,1),1);
       
       % Transform mean and std for lognormal
       [mu, sigma] = logtrans(PV2AXOmean,PV2AXOstd);
       weights(incomingconns>=bask(1) & incomingconns<=bask(2),1) = ...
           round(random('logn',mu,sigma,[1,size(incomingBASKconns,1)])',6);
       [mu, sigma] = logtrans(PN2AXOmean,PN2AXOstd);
       weights(incomingconns>=pyrA(1) & incomingconns<=pyrC(2),1) = ...
           round(random('logn',mu,sigma,[1,size(incomingPNconns,1)])',6);
       
       weightin{i+1} = outgoingvec(weights,i);
   end
   
   if i>=bask(1) && i<=bask(2) 
       %Find connections
       num_PN = ceil(PN2FSI*size(possible_PN_ID,1));
       num_BASK =  ceil(FSI2FSI*size(possible_BASK_ID,1));
       incomingPNconns = sortrows(datasample(stream,possible_PN_ID,num_PN,'Replace',false));
       incomingBASKconns = sortrows(datasample(stream,possible_BASK_ID,num_BASK,'Replace',false));
       %incomingAXOconns = incomingINTconns(incomingINTconns>=axo(1) & incomingINTconns<=axo(2));
       %incomingPVconns = incomingINTconns(incomingINTconns>=bask(1) & incomingINTconns<=bask(2));
       
       incomingconns = cat(1,incomingPNconns,incomingBASKconns); %concatenate
       connin{i+1} = outgoingvec(incomingconns,i);
       
       % Now get axonal delay
       pos_incomingconns = pos(ismember(pos(:,1),incomingconns),:);
       dist = (pos_incomingconns(:,2:4)-repmat(cell_pos,size(pos_incomingconns,1),1)).^2;
       dist = sqrt(sum(dist,2)); % distance in microns
       delays = dist/cond_vel; % divide by conduction velocity
       delays(delays<0.1) = 0.1; % set minimum delay (should be > dt in NEURON)
       
       delayin{i+1} = outgoingvec(delays,i);
       
       % Finally, assign weights
       weights = zeros(size(incomingconns,1),1);
       
       [mu, sigma] = logtrans(PV2PVmean,PV2PVstd);
       weights(incomingconns>=bask(1) & incomingconns<=bask(2),1) = ...
           round(random('logn',mu,sigma,[1,size(incomingBASKconns,1)])',6);
       [mu, sigma] = logtrans(PN2PVmean,PN2PVstd);
       weights(incomingconns>=pyrA(1) & incomingconns<=pyrC(2),1) = ...
           round(random('logn',mu,sigma,[1,size(incomingPNconns,1)])',6);
       
       weightin{i+1} = outgoingvec(weights,i);
       
       num_GAP = floor(GAPCONN*size(incomingBASKconns,1));
       gaps{i+1}(:,1) = repmat(i,num_GAP,1);
       gaps{i+1}(:,2) = sortrows(datasample(stream,incomingBASKconns,num_GAP,'Replace',false))';
       
       
   end
   
end

gaps_mat = cell2mat(gaps);

ids1=[num_cells+1:1:num_cells+size(gaps_mat,1)]';
ids2=[num_cells+2+size(gaps_mat,1):1:num_cells+1+2*size(gaps_mat,1)]';

gaps_mat(:,3) = ids1;
gaps_mat(:,4) = ids2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Write the output files.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sprintf('writing output files')

% gaps output file

fileID = fopen('gapconnections.dat','w');

for i=1:size(gaps_mat,1)
  fprintf(fileID,'%d\t%d\t%d\t%d',gaps_mat(i,1),gaps_mat(i,2),gaps_mat(i,3),gaps_mat(i,4));
  fprintf(fileID,'\n');
end

fclose(fileID);

% Position output file

fileID = fopen('position.dat','w');
fprintf(fileID,'%d\n',num_cells); %first line should be number of cells

for i=0:num_cells-1
  fprintf(fileID,'%d\t%d\t%d',pos(i+1,2),pos(i+1,3),pos(i+1,4));
  fprintf(fileID,'\n');
end

fclose(fileID);


connin = cell2mat(connin);
weightin = cell2mat(weightin);
delayin = cell2mat(delayin);

%
%% Connectivity output file

tic

fileID1 = fopen('connectivity.dat','w');
fprintf(fileID1,'%d\n',num_cells); %first line should be number of cells

fileID2 = fopen('weights.dat','w');
fprintf(fileID2,'%d\n',num_cells); %first line should be number of cells

fileID3 = fopen('delays.dat','w');
fprintf(fileID3,'%d\n',num_cells); %first line should be number of cells

sprintf('made fileIDs')   
for i=0:num_cells-1


	fprintf(fileID1,'%d\t',connin(i+1,~isnan(connin(i+1,1:end-1)))); 
	fprintf(fileID1,'%d',connin(i+1,~isnan(connin(i+1,end))));
	fprintf(fileID1,'\n');

	fprintf(fileID2,'%d\t',weightin(i+1,~isnan(weightin(i+1,1:end-1))));
	fprintf(fileID2,'%d',weightin(i+1,~isnan(weightin(i+1,end))));
	fprintf(fileID2,'\n');

	fprintf(fileID3,'%d\t',delayin(i+1,~isnan(delayin(i+1,1:end-1))));
	fprintf(fileID3,'%d',delayin(i+1,~isnan(delayin(i+1,end))));
	fprintf(fileID3,'\n');

end
toc
fclose(fileID1);
fclose(fileID2);
fclose(fileID3);


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
delete(gcp('nocreate'))
 
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
