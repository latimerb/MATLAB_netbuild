function flipped = flipmatwgt(orig_conn,orig_wgt,i)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % INPUTS: orig - original matrix to be flipped
    %         new_empty - empty matrix where new values will go
    
    % OUTPUT: flipped - same matrix but with post-synaptic cell as rows.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    inds = find(orig_conn(:,3:end)==i);
    
    flipped = NaN(1,4000);
    % These indices are numbers from
    % 1 to (size(connout,1)*size(connout,2)), numbered down the columns
    a = orig_wgt(:,3:end);
    flipped(1,1) = size(inds,1);
    flipped(1,2) = i;
    flipped(1,3:size(inds,1)+2) = ...
        mod(a(inds),size(orig_wgt,1))';
    
end
