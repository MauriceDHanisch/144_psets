from sympy import symbols, simplify

p = symbols('p', real=True)
k = symbols('k', integer=True)

# User's potential formula
# "p 2-p" -> (2-p)/p
# "-(1-p) k-1" -> -(1-p)**(k-1)
# "[ ... + k(1+ p 2(1-p) ) ]" -> k * (1 + 2*(1-p)/p) ??

term1 = (2 - p) / p
term2 = (1 - p)**(k - 1)
inside_bracket = (2 - p)/p + k * (1 + (2*(1-p))/p)

target = term1 - term2 * inside_bracket

# Calculated value (from previous output)
# We know the previous one was correct, let's just use the sum definition to be sure
from sympy import Sum
l = symbols('l', integer=True)
Pl = l * p**2 * (1 - p)**(l - 1)
Pk = (1 - p)**(k - 1) * (1 - p + k * p)
EL = Sum(l * Pl, (l, 1, k - 1)).doit() + k * Pk

diff = (EL - target).simplify()
print(f"Difference: {diff}")
if diff == 0:
    print("Match confirmed!")
else:
    print("No match.")

# Let's try to pretty print the target to see if it looks like what the user pasted
from sympy import pprint
print("\nPretty Print:")
pprint(target)
