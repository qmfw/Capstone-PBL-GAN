def print_msg(msg):
    # This is the outer enclosing function

    def printer():
        # This is the nested function
        print(msg)

    printer()

# We execute the function
# Output: Hello
print_msg("Hello")




def psg(msg):
    # This is the outer enclosing function

    def pnt():
        # This is the nested function
        print(msg)

    return pnt  # returns the nested function

# Now let's try calling this function.
# Output: Hello
another = psg("Hello2")
another()




def make_multiplier_of(n):
    def multiplier(x):
        return x * n
    return multiplier

# Multiplier of 3
times3 = make_multiplier_of(3)

# Multiplier of 5
times5 = make_multiplier_of(5)

# Output: 27
print(times3(9))

# Output: 15
print(times5(3))

# Output: 30
print(times5(times3(2)))




def inc(x):
    return x + 1

def dec(x):
    return x - 1

def operate(func, x):
    result = func(x)
    return result

# 4
operate(inc,3)
# 2
operate(dec,3)

