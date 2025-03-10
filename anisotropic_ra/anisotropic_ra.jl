using LinearAlgebra, Statistics
using JuMP, SCS

function run_rotation_averaging(Rrel, H; anisotropic_cost=true, cSO3_constraints=true, max_iters=500_000, eps_abs=1.0e-5, eps_rel=1.0e-6, eps_infeas=1.0e-8)
    """
    Args:
        Rrel -- 3Nx3N matrix of relative rotations (unobserved blocks are expected to be all zeros)
        H -- 3Nx3N matrix of Hessians
    Returns:
        R_est -- Nx3x3 matrix of estimated absolute rotations extracted from the solution
        stat -- status of optimization
        stime -- solver runtime
        rk -- rank of the solution
        obj_val -- objective value at the solution
    """
    cost_matrix = construct_cost_matrix(Rrel, H; anisotropic=anisotropic_cost)

    N = Rrel.size[1] ÷ 3
    I3x3 = [1.0  0  0; 0  1.0  0; 0  0  1.0]
    
    model = Model(SCS.Optimizer)
    set_attribute(model, "eps_abs", eps_abs)
    set_attribute(model, "eps_rel", eps_rel)
    set_attribute(model, "eps_infeas", eps_infeas)
    set_attribute(model, "max_iters", max_iters)
    set_silent(model)
    @variable(model, X[1:3*N, 1:3*N], PSD)

    # Add block diagonal constraints
    for i = 1:N
        @constraint(model, X[3*i-2:3*i, 3*i-2:3*i] == I3x3)
    end

    # Add conv(SO(3)) constraints
    if cSO3_constraints
        for i=1:N
            for j=i+1:N
                i0 = 3*(i-1); j0 = 3*(j-1)
                if sqrt(sum(Rrel[i0+1:i0+3, j0+1:j0+3].^2)/9) > 1e-7
                    Yk = @variable(model, [1:4,1:4], PSD)

                    @constraint(model, 1 - X[i0+1,j0+1] - X[i0+2,j0+2] + X[i0+3,j0+3] - Yk[1,1] == 0)
                    @constraint(model, 1 + X[i0+1,j0+1] - X[i0+2,j0+2] - X[i0+3,j0+3] - Yk[2,2] == 0)
                    @constraint(model, 1 + X[i0+1,j0+1] + X[i0+2,j0+2] + X[i0+3,j0+3] - Yk[3,3] == 0)
                    @constraint(model, 1 - X[i0+1,j0+1] + X[i0+2,j0+2] - X[i0+3,j0+3] - Yk[4,4] == 0)

                    @constraint(model, X[i0+1,j0+3] + X[i0+3,j0+1] - Yk[1,2] == 0)
                    @constraint(model, X[i0+1,j0+2] - X[i0+2,j0+1] - Yk[1,3] == 0)
                    @constraint(model, X[i0+2,j0+3] + X[i0+3,j0+2] - Yk[1,4] == 0)

                    @constraint(model, X[i0+2,j0+3] - X[i0+3,j0+2] - Yk[2,3] == 0)
                    @constraint(model, X[i0+1,j0+2] + X[i0+2,j0+1] - Yk[2,4] == 0)

                    @constraint(model, X[i0+3,j0+1] - X[i0+1,j0+3] - Yk[3,4] == 0)
                end
            end
        end
    end

    @objective(model, Max, sum(cost_matrix .* X))
    optimize!(model)
    stime = solve_time(model)
    obj_val = objective_value(model)
    stat = is_solved_and_feasible(model; dual=true) ? "solved and feasible" : raw_status(model)

    X_res = value(X)
    rk = rank(X_res)
    
    # Project solution into rank-3 matrix
    if rk > 3
        U, S, V = svd(X_res)
        S[4:end] .= 0.0
        X_res = U * diagm(S) * V'
    end

    # Extract rotations from the solution X
    R_est = zeros(N, 3, 3)
    for i = 1:N
        R_est[i,:,:] .= project_on_SO3(X_res[3*i-2:3*i, 1:3])
    end
    return R_est, stat, stime, rk, obj_val
end

function construct_cost_matrix(Rrel, H; anisotropic=true, rescale_hessians=true)
    """
    Args:
        Rrel -- 3Nx3N matrix of relative rotations (unobserved blocks are expected to be all zeros)
        H -- 3Nx3N matrix of Hessians
    Returns:
        cost_matrix -- 3Nx3N symmetric cost matrix for rotation averaging
    """
    N = Rrel.size[1] ÷ 3 # number of cameras/poses
    K = N*(N-1) ÷ 2 # number of relative rotations
    I3x3 = [1.0  0  0; 0  1.0  0; 0  0  1.0]

    # Collect eigenvalues of all Hessians for re-scaling
    H_eigvals = zeros(K,3)
    k = 1
    for i=1:N
        for j=i+1:N
            ii, jj = 3*i-2:3*i, 3*j-2:3*j
            if sqrt(sum(Rrel[ii, jj].^2)/9) > 1e-7
                H_eigvals[k,:] .= eigvals(H[ii, jj])
                k += 1
            end
        end
    end
    K_observed = k-1
    H_eigvals = H_eigvals[1:K_observed,:]

    # Make isotropic/anisotropic cost matrix
    cost_matrix = zeros(3*N, 3*N)
    for i=1:N
        for j=i+1:N
            ii, jj = 3*i-2:3*i, 3*j-2:3*j
            Rrel_ij = Rrel[ii, jj]
            if sqrt(sum(Rrel_ij.^2)/9) > 1e-7
                H_ij = H[ii, jj]
                if rescale_hessians
                    H_ij ./= mean(maximum(H_eigvals, dims=2))
                end
                M_ij = tr(H_ij) / 2 * I3x3 - H_ij
                cost_matrix[ii, jj] .= anisotropic ? (M_ij * Rrel_ij)' : Rrel_ij'
            end
        end
    end
    cost_matrix .= cost_matrix + cost_matrix'
    return cost_matrix
end

function project_on_SO3(M)
    U, _, V = svd(M)
    R = U * V'
    if sign(det(R)) < 0
        R .= U * Diagonal([1,1,-1]) * V'
    end
    return R
end

function rank(X; threshold=0.999)
    _, singular_vals, _ = svd(X)
    return findfirst(cumsum(singular_vals) / sum(singular_vals) .> threshold)
end