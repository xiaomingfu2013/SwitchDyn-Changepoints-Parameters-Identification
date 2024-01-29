using Test

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "ChangePointCodeRepo tests" begin
    @info "test adjoint method"
    include("test_adjoint_method.jl")
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60; digits=3), " minutes")
