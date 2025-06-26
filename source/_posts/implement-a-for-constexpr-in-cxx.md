---
title: Implement a For Constexpr in C++
date: 2024-12-22 14:47:55
tags: C++
description: C++ 17 introduced a new keyword if constexpr. While we may need a for constexpr, however, there are no such keywords provided. This article is a guide on how to implement such a feature with C++ template.
---

# Introduction
C++ 17 introduced a new feature, namely `if constexpr`. Unlike prior `if` statements, `if constexpr` is evaluated at compile time and the body will be compiled only when the condition evaluates to true. In other words, the body of `if constexpr` statement will not be compiled and will be discarded when the condition evaluates to false. This feature requires that the condition of `if constexpr` is indeed a `constexpr`. Here is an example.
```C++
if constexpr (true) {
    std::cout << "true"; // compiled
} else {
    std::cout << "false"; // not compiled
}
```
The body of `else` will not be compiled and will be discarded at compile time.

In some cases, we may want to use `if constexpr` in a `for` loop to reduce branch statements and speed up the runtime. However, C++ standards do not support `constexpr` loop variable, although obviously they should be when the range of `for` loop is determined by constants. 
```C++
for (int i = 0; i < 10; i++) {
    if constexpr (i != 0) { // error!
        do_something();
    }
}
``` 

In the following parts of this article, I will present a method to implement a equivalent of `for constexpr`.

# Some Template Functions
Before talking about the real topic, let's first see some template functions. If you are already familiar with them, you can skip to the next section.

## std::forward
Suppose we have function `foo` that accepts a r-value reference. We have a overloaded function `print_ref_type` that would determine the reference type.
```C++
#include <iostream>
#include <utility>

template <typename _Tp>
constexpr void print_ref_type(_Tp&& var) {
    std::cout << "r-value reference";
}

template <typename _Tp>
constexpr void print_ref_type(_Tp& var) {
    std::cout << "l-value reference";
}

template <typename _Tp>
constexpr void foo(_Tp&& var) {
    print_ref_type(var);
}

int main () {
    foo(1);
}
```

Running this piece of code gives
```shell
l-value reference
```

How? The function `foo` is indeed accepting a `r-value`! The problem is that, in the body of function `foo`, `var` is a named variable thus considered a `l-value`. That's why `std::forward` comes, which can cast a l-value reference into a r-value reference. The implementation of `std::forward` is shown below. We can see that it accepts `l-value` reference after removing the reference of `_Tp` and getting the actual type. Then, the `l-value` reference is cast into a `r-value` reference.
```C++
  template<typename _Tp>
    _GLIBCXX_NODISCARD
    constexpr _Tp&&
    forward(typename std::remove_reference<_Tp>::type& __t) noexcept
    { return static_cast<_Tp&&>(__t); }
```

Adding the `std::forward` into our previous example finally creates the expected behavior.
```C++
#include <iostream>
#include <utility>

template <typename _Tp>
constexpr void print_ref_type(_Tp&& var) {
    std::cout << "r-value reference";
}

template <typename _Tp>
constexpr void print_ref_type(_Tp& var) {
    std::cout << "l-value reference";
}

template <typename _Tp>
constexpr void foo(_Tp&& var) {
    print_ref_type(std::forward<_Tp>(var));
}

int main () {
    foo(1);
}
```
```shell
r-value reference
```

## std::integral_constant
This feature allows us to represent a `constexpr` as a type and pass it into a function. If a function accepts a variable and we passed a `constexpr` into the function, the parameter may lose the attribute `constexpr`. std::integral_constant guarantees that a constant will still be a constant after being passed into a function. Here is an example.
```C++
#include <iostream>
#include <utility>

constexpr void test(auto var) {
    if constexpr (var) {
        std::cout << "ok";
    }
}

int main (){
    test(true);
}
```

Compiling this piece of code will generate an error.
```shell
test.cpp: In instantiation of 'constexpr void test(auto:1) [with auto:1 = bool]':
test.cpp:10:9:   required from here
test.cpp:4:5: error: 'var' is not a constant expression
    4 |     if constexpr (var) {
      |     ^~
```
Using std::integral_constant should reduce this error.
```C++
#include <iostream>
#include <utility>

constexpr void test(auto var) {
    if constexpr (var) {
        std::cout << "ok";
    }
}

int main (){
    test(std::integral_constant<bool, true>());
}
```
```shell
ok
```


# Our For Constexpr
Finally comes our `for constexpr` equivalent. Now the `if constexpr` should work in our `for constexpr` loop. Note that the body of loop is passed as a `lambda expression` into the function `for_constexpr`. The lambda expression accepts a loop index and the index can be used a constant in the loop body.
```C++
#include <iostream>
#include <utility>

template <int Start, int End, int Step = 1, typename F>
constexpr void for_constexpr(F&& f) {
    if constexpr (Start < End) {
        f(std::integral_constant<int, Start>());
        for_constexpr<Start + Step, End, Step>(std::forward<F>(f));
    }
}

int main () {
    for_constexpr<1, 5>([&](auto i) {
        if constexpr (i > 3) {
            std::cout << i << std::endl;
        }
    });
}
```

Some compilers may restrict template recursion depth and refuse to compile the code. We may need to specify a flag `-ftemplate-depth` when compiling the code.
```shell
g++ your_code.cpp -ftemplate-depth=1000 -o your_bin
```
