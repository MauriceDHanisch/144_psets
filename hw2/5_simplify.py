from sympy import symbols, Sum, simplify, limit, oo, factor, collect, pprint, latex

# Define the variables
k = symbols('k', integer=True, positive=True)
p = symbols('p', real=True, positive=True)
l = symbols('l', integer=True, positive=True)

print("--- 1. Verification of P(l=k) ---")

# P(l) for case l < k
Pl = l * p**2 * (1 - p)**(l - 1)

# Sum P(l) from 1 to k-1
sum_probs = Sum(Pl, (l, 1, k - 1)).doit().simplify()
Pk_calculated = (1 - sum_probs).simplify()

# Target Formula: (1-p)^(k-1) * (1 - p + k*p)
Pk_formula = (1 - p)**(k - 1) * (1 - p + k * p)

print("Target Formula:", Pk_formula)
print("Calculated:", Pk_calculated)

# Check equivalence by dividing
ratio = (Pk_calculated / Pk_formula).simplify()
print(f"Ratio (should be 1): {ratio}")

print("\n--- 2. Expected Value E[L] ---")
# E[L] = Sum_{l=1}^{k-1} l * P(l) + k * P(k)

E_L_sum = Sum(l * Pl, (l, 1, k-1)) + k * Pk_formula
exact_result = E_L_sum.doit().simplify()

# We collect terms that share (1-p)**(k-2)
# We also simplify the chunks inside the brackets
readable_result = collect(exact_result, (1-p)**(k-2), func=simplify)

print("Readable Form:")
print(readable_result)

print("Latex Representation:")
print(latex(readable_result))