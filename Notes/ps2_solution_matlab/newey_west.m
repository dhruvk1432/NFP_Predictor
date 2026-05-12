function lrv = newey_west(Y)

    % Newey-West HAC estimator

    T = size(Y,1);
    bw = ceil(0.75*T^(1/3)); % Default bandwidth

    lrv = Y'*Y;
    for l=1:bw-1
        weight = 1-l/bw;
        aux = Y(l+1:end,:)'*Y(1:end-l,:);
        lrv = lrv + weight*(aux+aux');
    end
    lrv = lrv/T;

end