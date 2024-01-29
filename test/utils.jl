import Base.Cartesian: @nexprs, @ntuple
"""
    split according to the length of the tuple
    Input:
        x: vector
        len_tuple: tuple of length n,
    Example:
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tuple = (2, 3, 4)
    Return Arguments:
        X = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
"""
function split_vector(x::AbstractVector{T}, len_tuple::Tuple{Vararg{Int}}) where {T}
    return split_vector(x, len_tuple, Val(length(len_tuple)))
end

@generated function split_vector(
    x::AbstractVector{T}, len_tuple::Tuple{Vararg{Int}}, ::Val{N}
) where {T,N}
    quote
        len = 0
        @nexprs $N (n) -> (s_n = len + 1; e_n = len + len_tuple[n]; len = e_n)
        X = @ntuple $N (n) -> (reshape(view(x, s_n:e_n), len_tuple[n]))
        return X
    end
end
