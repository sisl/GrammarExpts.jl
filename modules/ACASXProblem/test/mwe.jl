module A
export foo, bar
export to_function

f(x) = 0
g(x) = x + 1

function to_function(expr)
    @eval f(x) = $expr
    f
end
function foo(expr)
    f = to_function(expr)
    for x=1:1000
       f(x)
    end
    f
end
function bar(expr)
    f = to_function(expr) 
    for x=1:1000
        g(x)
    end
    f
end
end #module

using A

@time expr = :(x+1)
@time to_function(expr)
A.f(1)
foo(expr)
bar(expr)

for i=1:1000000
    println("$i: foo-bar")
    i2 = rand(1:i)
    i3 = rand(1:i) 
    i4 = rand(1:i)
    i5 = rand(1:i)
    i6 = rand(1:i) 
    i7 = rand(1:i)
    i8 = rand(1:i)
    expr = :(x+$i+$i2+$i3+$i4+$i5+$i6+$i7+$i8)
    @time foo(expr)
    @time bar(expr)
end
