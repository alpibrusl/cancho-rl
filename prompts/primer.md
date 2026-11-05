# cancho, for small integer functions

cancho is a small systems language. A program for these tasks is a file that starts with `edition 7;` followed by functions.

```cancho
edition 7;

fn clamp(x: int, lo: int, hi: int) -> [] int {
    if x < lo { return lo; }
    if x > hi { return hi; }
    return x;
}

fn is_leap_year(y: int) -> [] bool {
    if y % 400 == 0 { return true; }
    if y % 100 == 0 { return false; }
    return y % 4 == 0;
}

fn gcd(a: int, b: int) -> [] int {
    var x = a;
    var y = b;
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    return x;
}
```

Rules that matter:

- Types are `int` (64-bit) and `bool`. The `[]` after `->` is the effect list; it is always empty here.
- `let` binds once; `var` can be assigned again with `=`. Every statement ends with `;`. Conditions take no parentheses and bodies always take braces.
- Loops are `while cond { ... }`. There is no `for`. Recursion is allowed.
- Operators: `+ - * / %`, comparisons, `&& || !`. `/` truncates toward zero and `%` keeps the sign of the left side, so `-7 % 2` is `-1`: compare a remainder with `== 0`, nothing else, when the sign can be negative.
- Write negation as `0 - x`. Integer overflow is an error, not a wrap: keep intermediate values inside 64 bits.
- A function that uses a helper must define the helper too, above or below it.
- There is no `+=`, `-=`, `*=` or `/=`: write `x = x + 1;`. There is no `abs`, `min` or `max`: write an `if`. Function parameters cannot be assigned: copy one into a `var` first.
