def my_sum(*args):
    return sum(args)

print(my_sum(1, 2, 3, 4))
# 10

print(my_sum(1, 2, 3, 4, 5, 6, 7, 8))
# 36




def my_sum2(*args):
    print('args: ', args)
    print('type: ', type(args))
    print('sum : ', sum(args))

my_sum2(1, 2, 3, 4)
# args:  (1, 2, 3, 4)
# type:  <class 'tuple'>
# sum :  10




def func_args(arg1, arg2, *args):
    print('arg1: ', arg1)
    print('arg2: ', arg2)
    print('args: ', args)

func_args(0, 1, 2, 3, 4)
# arg1:  0
# arg2:  1
# args:  (2, 3, 4)

func_args(0, 1)
# arg1:  0
# arg2:  1
# args:  ()




def func_args2(arg1, *args, arg2):
    print('arg1: ', arg1)
    print('arg2: ', arg2)
    print('args: ', args)

# func_args2(0, 1, 2, 3, 4)
# TypeError: func_args2() missing 1 required keyword-only argument: 'arg2'

func_args2(0, 1, 2, 3, arg2=4)
# arg1:  0
# arg2:  4
# args:  (1, 2, 3)




def func_args_kw_only(arg1, *, arg2):
    print('arg1: ', arg1)
    print('arg2: ', arg2)

# func_args_kw_only(100, 200)
# TypeError: func_args_kw_only() takes 1 positional argument but 2 were given

func_args_kw_only(100, arg2=200)
# arg1:  100
# arg2:  200




def func_kwargs(**kwargs):
    print('kwargs: ', kwargs)
    print('type: ', type(kwargs))

func_kwargs(key1=1, key2=2, key3=3)
# kwargs:  {'key1': 1, 'key2': 2, 'key3': 3}
# type:  <class 'dict'>




def func_kwargs_positional(arg1, arg2, **kwargs):
    print('arg1: ', arg1)
    print('arg2: ', arg2)
    print('kwargs: ', kwargs)

func_kwargs_positional(0, 1, key1=1)
# arg1:  0
# arg2:  1
# kwargs:  {'key1': 1}




d = {'key1': 1, 'key2': 2, 'arg1': 100, 'arg2': 200}

func_kwargs_positional(**d)
# arg1:  100
# arg2:  200
# kwargs:  {'key1': 1, 'key2': 2}

# def func_kwargs_error(**kwargs, arg):
#     print(kwargs)

# SyntaxError: invalid syntax
