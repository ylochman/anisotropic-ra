using Random, LinearAlgebra

include("helpers.jl")

function generate_data(N, p; min_H_eigval=10, max_H_eigval=100)
    """
    Args:
        N (int) -- number of cameras/poses
        p (float) -- fraction of observed data, p in (0,1]
        min_H_eigval/max_H_eigval -- eigenvalue range of the precision matrix (== Hessian)
    Returns:
        R -- Nx3x3 matrix of absolute rotations
        Rrel -- 3Nx3N matrix of relative rotations
        H -- 3Nx3N matrix of Hessians
    """
    K = N * (N-1) ÷ 2
    K_observed = max(N-1, Int(ceil(K * p) ÷ 1))
    I3x3 = [1.0  0  0; 0  1.0  0; 0  0  1.0]
    
    # Observation pattern
    W = zeros(N,N)
    observed_idx = randperm(K)[1:K_observed]
    triu_indices = findall(!iszero, triu(ones(N,N),1))
    W[triu_indices[observed_idx]] .= 1.0
    W .= W + W'
    while !(all(sum(W,dims=1).>0))
        W .= zeros(N,N)
        observed_idx = randperm(K)[1:K_observed]
        triu_indices = findall(!iszero, triu(ones(N,N),1))
        W[triu_indices[observed_idx]] .= 1.0
        W .= W + W'
    end

    # Absolute rotations
    R = zeros(N, 3, 3)
    for i=1:N
        R[i,:,:] .= exp(skewsym(randn(3,1)))
    end

    # Hessians
    H = zeros(3*N, 3*N)
    for i=1:N
        for j=i+1:N
            if W[i,j] > 0
                ii, jj = 3*i-2:3*i, 3*j-2:3*j
                eigvecs = svd(randn(3,3)).U;
                eigvals = rand(3) .* (max_H_eigval - min_H_eigval) .+ min_H_eigval
                H[ii, jj] = eigvecs * diagm(eigvals) * eigvecs'
            end
        end
    end

    # Relative rotations
    Rrel = zeros(3*N, 3*N)
    for i=1:N
        for j=i+1:N
            if W[i,j] > 0
                ii, jj = 3*i-2:3*i, 3*j-2:3*j
                Rrel_ij = R[j,:,:] * R[i,:,:]'
                H_ij = H[ii,jj]
                dw = sqrt(inv(H_ij)) * randn(3,1)
                Rrel[ii,jj] .= exp(skewsym(dw)) * Rrel_ij
            end
        end
    end

    return R, Rrel, H
end