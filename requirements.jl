import Pkg

add_packages = packages -> for pkg=packages Pkg.add(pkg) end

add_packages(["Random", "LinearAlgebra", "Statistics"])
add_packages(["JuMP", "SCS"])