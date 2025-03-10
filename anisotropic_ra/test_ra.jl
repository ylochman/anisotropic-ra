using Random, LinearAlgebra

include("data.jl")
include("helpers.jl")
include("anisotropic_ra.jl")

function test_rotation_averaging(N=50, p=0.5; min_H_eigval=10, max_H_eigval=100, anisotropic_cost=true, cSO3_constraints=true)
    """
    Args:
        N (int) -- number of cameras/poses
        p (float) -- fraction of observed data, p in (0,1]
        min_H_eigval/max_H_eigval -- eigenvalue range of the precision matrix (== Hessian)
        anisotropic_cost -- if true, uses the proposed anisotropic cost
        cSO3_constraints -- if true, includes the proposed convex hull constraints
    """
    R_true, Rrel, H = generate_data(N, p; min_H_eigval=min_H_eigval, max_H_eigval=max_H_eigval)
    R_est, stat, stime, rk, obj_val = run_rotation_averaging(Rrel, H; anisotropic_cost=anisotropic_cost, cSO3_constraints=cSO3_constraints)

    # Align the solution with GT
    M_align = zeros(3, 3)
    for i = 1:N
        M_align .+= R_est[i,:,:]' * R_true[i,:,:]
    end
    R_align = project_on_SO3(M_align)
    for i = 1:N
        R_est[i,:,:] .= R_est[i,:,:] * R_align
    end

    # Evaluate the solution
    fro_err = sqrt(sum((R_est - R_true).^2))
    mahal_err = mahalonobis_distance(R_est, R_true, H)
    angles = zeros(N)
    for i=1:N
        angles[i] = norm(axis_angle(R_true[i,:,:] * R_est[i,:,:]')) / pi * 180
    end 
    angular_err = rms(angles)
    display("Frobenius error: $fro_err, Mahalonobis error: $mahal_err, Angular error: $angular_err, Solver runtime: $stime")
end

test_rotation_averaging()