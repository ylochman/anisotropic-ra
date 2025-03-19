using Random, LinearAlgebra, MAT

include("helpers.jl")

function generate_data(N, p; min_H_eigval=10, max_H_eigval=100)
    """ Create synthetic camera graph with N cameras and K observed noisy relative rotations, where K = max(N-1, ⌈pN(N-1)/2⌉). Noise perturbations achieved by left-multiplying with exp([Δw]_x), where Δw ~ 𝒩(0, H^{-1}).
     Args:
        N (int) -- number of cameras/poses
        p (float) -- fraction of observed relative rotations, p in (0,1]
        min_H_eigval/max_H_eigval -- eigenvalue range of the precision matrices (== Hessians)
    Returns:
        R -- Nx3x3 matrix of absolute rotations
        Rrel -- 3Nx3N matrix of observed noisy relative rotations
        H -- 3Nx3N matrix of Hessians
    """
    K = N * (N-1) ÷ 2
    K_observed = max(N-1, Int(ceil(K * p) ÷ 1))
    
    # Observation pattern
    W = Bool.(zeros(N,N))
    observed_idx = randperm(K)[1:K_observed]
    triu_indices = findall(!iszero, triu(ones(N,N),1))
    W[triu_indices[observed_idx]] .= true
    W .= W + W'
    while !(all(sum(W,dims=1).>0))
        W .= Bool.(zeros(N,N))
        observed_idx .= randperm(K)[1:K_observed]
        triu_indices .= findall(!iszero, triu(ones(N,N),1))
        W[triu_indices[observed_idx]] .= true
        W .= W + W'
    end

    # Absolute rotations
    R = zeros(N, 3, 3)
    for i=1:N
        R[i,:,:] .= exp(skewsym(normalize(rand(3))*rand()*2*pi))
    end

    # Hessians and corresponding relative rotations
    Rrel = zeros(3*N, 3*N)
    H = zeros(3*N, 3*N)
    for i=1:N
        for j=i+1:N
            if W[i,j] > 0
                ii, jj = 3*i-2:3*i, 3*j-2:3*j

                # Hessian
                eigvecs = exp(skewsym(normalize(rand(3))*rand()*2*pi))
                eigvals = rand(3) .* (max_H_eigval - min_H_eigval) .+ min_H_eigval
                H[ii, jj] .= eigvecs * diagm(eigvals) * eigvecs'
                
                # Perturbed relative rotation
                dw = eigvecs * (randn(3,1) ./ sqrt.(eigvals))
                Rrel_ij = R[j,:,:] * R[i,:,:]'
                Rrel[ii,jj] .= exp(skewsym(dw)) * Rrel_ij
            end
        end
    end

    return R, Rrel, H
end

function read_matlab_data(matfile)
    data = matread(matfile)
    N = data["Rgt"].size[1] ÷ 3 # number of cameras/poses
    I3x3 = [1.0  0  0; 0  1.0  0; 0  0  1.0]
    
    # Absolute rotations
    R_true = zeros(N, 3, 3)
    for i=1:N
        R_true[i,:,:] = data["Rgt"][3*(i-1)+1:3*(i-1)+3,:]
    end

    # Hessians (extracted from matrices M_{ij})
    H = zeros(3*N, 3*N)
    for i=1:N
        for j=i+1:N
            i0 = 3*(i-1); j0 = 3*(j-1)
            Rrel_ij = data["Rrel"][i0+1:i0+3, j0+1:j0+3]
            if sum(Rrel_ij.^2) > 0
                Mij = data["Alpha"][i0+1:i0+3, j0+1:j0+3]
                Hij = tr(Mij) * I3x3 - Mij
                H[i0+1:i0+3, j0+1:j0+3] = Hij
            end
        end
    end

    return R_true, data["Rrel"], H
end