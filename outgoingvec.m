function connvecout=outgoingvec(outgoingconns,i)
    connvecout = NaN(1,4000);
    connvecout(1,1) = size(outgoingconns,1); %number of connections in row
    connvecout(1,2) = i; %GID of presynaptic cell
    connvecout(1,3:size(outgoingconns,1)+2) = outgoingconns';
end