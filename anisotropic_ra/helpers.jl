using Random, LinearAlgebra

function skewsym(t)
    return [  0  -t[3]  t[2];
             t[3]  0   -t[1];
            -t[2] t[1]   0  ]
end

function axis_angle(R)
    logR = log(R)
    return [logR[3,2], logR[1,3], logR[2,1]]
end

function project_on_SO3(M)
    U, _, V = svd(M)
    R = U * V'
    if sign(det(R)) < 0
        R .= U * Diagonal([1,1,-1]) * V'
    end
    return R
end

function mahalonobis_distance(R1, R2, H)
    N = R1.size[1]
    error = 0
    for i=1:N
        ii = 3*i-2:3*i
        Hi = sum([H[ii, 3*j-2:3*j] for j=1:N if j!=i])
        wi_1 = axis_angle(R1[i,:,:])
        wi_2 = axis_angle(R2[i,:,:])
        dw_i1 = wi_1 - wi_2
        dw_i2 = wi_1 + wi_2
        error += min(dw_i1' * Hi * dw_i1, dw_i2' * Hi * dw_i2)
    end
    return sqrt(error)
end

function rms(diff)
    return sqrt(mean((diff[:]).^2))
end