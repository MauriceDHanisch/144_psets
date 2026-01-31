# %% [markdown]
# # HW 3 Problem 1.1: Heavy vs. Light Tails

# %%
import numpy as np
from scipy import stats
from scipy.special import gamma
from sklearn import linear_model
import matplotlib.pyplot as plt
import heapq

n = 10000 # Arbitrary plotting value for large n

# %% [markdown]
# ## Part a: Law of Large Numbers
# 
# Make two plots of $S_n$ vs. $n$ for each of the distributions - the first plot over $n \\in$ {1,2, . . . ,20} and the second one over the full range of $n$. **Interpret your plots, in light of the law of large numbers and write your analysis here.**

# %%
import os

def make_graph_a(
    xs,
    ys,
    distribution="Standard Normal",
    xlabel="Number of Variable Draws (n)",
    ylabel="Cumulative Sum"):
    """
    xs: List of x values to plot
    ys: List of y values to plot
    distribution: The name of the distribution that you are plotting
    """
    title = "{0} of {1} Distribution".format(ylabel, distribution)
    
    plt.subplots(1, 2, figsize=(15, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xs[:20], ys[:20])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + " (First 20 values)")

    plt.subplot(1, 2, 2)
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if not os.path.exists('../latex/figs'):
        os.makedirs('../latex/figs')
    
    filename = f"plot_1.1_a_{distribution.replace(' ', '_')}.png"
    plt.savefig(f"../latex/figs/{filename}")

    plt.show()

# %%
normal_draws = np.random.normal(loc=1.0, scale=1.0, size=n)

alpha_weibull = 0.3
scale_weibull = 1.0 / gamma(1.0 + 1.0/alpha_weibull)
weibull_draws = scale_weibull * np.random.weibull(a=alpha_weibull, size=n)

alpha_pareto = 0.5
xm_pareto = 1.0 / 3.0
pareto_draws = stats.pareto.rvs(b=alpha_pareto, scale=xm_pareto, size=n)

x_range = np.linspace(1, n, num=n)
normal_cumsum = np.cumsum(normal_draws)
weibull_cumsum  = np.cumsum(weibull_draws)
pareto_cumsum = np.cumsum(pareto_draws)

# %%
make_graph_a(
    x_range,
    normal_cumsum,
    distribution="Standard Normal")

# %%
make_graph_a(
    x_range,
    weibull_cumsum,
    distribution="Weibull")

# %%
make_graph_a(
    x_range,
    pareto_cumsum,
    distribution="Pareto")

# %% [markdown]
# ## Part b: Central Limit Theorem
# 
# The Central Limit Theorem tells us that deviations of $S_n$ from its mean are of size $\sqrt{n}$. That is, $S_n \approx nE[X] + O(\sqrt{n})$. Plot $\frac{S_n - nE[X]}{\sqrt{n}}$ vs. $n$ for each of the distributions. **Interpret your plots, in light of the central limit theorem, and write your analysis here. Why aren't we also testing the Pareto distribution here?**

# %%
mu = 1.0

normal_clt = (normal_cumsum - x_range * mu) / np.sqrt(x_range)
weibull_clt = (weibull_cumsum - x_range * mu) / np.sqrt(x_range)

plt.subplots(1, 2, figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(x_range, normal_clt)
plt.xlabel("Number of Variable Draws (n)")
plt.ylabel("Scaled Deviation of Sn from Mean")
plt.title("Central Limit Theorem (Normal)")

plt.subplot(1, 2, 2)
plt.plot(x_range, weibull_clt)
plt.xlabel("Number of Variable Draws (n)")
plt.ylabel("Scaled Deviation of Sn from Mean")
plt.title("Central Limit Theorem (Weibull)")

if not os.path.exists('../latex/figs'):
    os.makedirs('../latex/figs')
plt.savefig("../latex/figs/plot_1.1_b_CLT.png")

plt.show()

# %%
# Calculate Variance for Weibull (alpha=0.3, mean=1)
alpha_w = 0.3
beta_w = 1.0 / gamma(1 + 1.0/alpha_w)
variance_weibull = beta_w**2 * (gamma(1 + 2.0/alpha_w) - gamma(1 + 1.0/alpha_w)**2)
std_weibull = np.sqrt(variance_weibull)

print(f"Weibull Variance: {variance_weibull:.2f}")
print(f"Weibull Std Dev: {std_weibull:.2f}")


# %% [markdown]
# ## Part c: The 80-20 rule
# 
# Vilfredo Pareto was motivated to define the Pareto distribution by this observation: 80% of the wealth in society is held by 20% of the population. This is an important distinguishing feature between heavy-tailed and light-tailed distributions. To observe this, suppose that your samples represent the incomes of 10000 individuals in a city. Since some of your samples for the Normal distribution might be negative, ignore the case of the Normal distribution for this part of the problem, since a negative income doesn't make much sense. Compute the fraction $f(r)$ of the total income of the city held by the wealthiest $r$% of the population, for $r$ = 1,2,...,20. For each of the distributions, plot $f(r)$ vs. $r$. (preferably both functions on a single plot). **Interpret your plot(s) and write your analysis here**.

# %%
def get_frac_wealth(draws, r_range):
    total_wealth = np.sum(draws)
    fracs = []
    for r in r_range:
        num_top = int((r / 100.0) * len(draws))
        if num_top == 0:
            fracs.append(0)
            continue
        largest = heapq.nlargest(num_top, draws)
        fracs.append(np.sum(largest) / total_wealth)
    return fracs

rRange = np.linspace(1, 20, num=20)
weibull_largest = get_frac_wealth(weibull_draws, rRange)
pareto_largest = get_frac_wealth(pareto_draws, rRange)

plt.figure()
plt.plot(rRange, weibull_largest, 'r', label="Weibull")
plt.plot(rRange, pareto_largest, 'b', label="Pareto")
plt.xlabel("Wealthiest r% of population.")
plt.ylabel("Percent of total income.")
plt.title("80/20 rule for Weibull and Pareto Distributions")
plt.legend()

if not os.path.exists('../latex/figs'):
    os.makedirs('../latex/figs')
plt.savefig("../latex/figs/plot_1.1_c_80_20.png")

plt.show()

# %% [markdown]
# ## Part d: Identifying Heavy Tails
# 
# For each of the distributions (i)-(iii), plot the frequencies and ranks of the 10000 samples on log-log scales, using separate plots for each distribution. Since we are using a log-log scale, filter out all negative and zero values before graphing. For the frequency plots, remember to experiment with various binsizes and to choose one such that the plots are useful. (Note that bins aren't needed for the rank plot.) Then, use linear regression to fit a line through the points on each plot. Display the best-fit lines on the plots as well as the R-squared values. What do your plots tell you about identifying heavy tails based on frequency and rank plots? **Interpret your plot(s) and write your analysis here**.

# %%
def pdf(data, bins=50):
    '''Takes an array with random samples from a distribution, 
    and creates an approximate PDF of points.
    Returns a tuple of two vectors x, y where 
    y_i = P(x_i - dx/2 <= data < x_i + dx/2) / dx'''
  
    min_val = np.min(data)
    max_val = np.max(data)
    
    if min_val <= 0:
        min_val = np.min(data[data > 0])
        
    bins_array = np.logspace(np.log10(min_val), np.log10(max_val), bins)
    
    hist, bin_edges = np.histogram(data, bins=bins_array, density=True)
    
    x = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    y = hist
    
    mask = y > 0
    return x[mask], y[mask]
    
def ccdf(data):
    '''Takes an array with random samples from a distribution, 
    and creates an approximate CCDF (complementary CDF) of points. 
    Returns a tuple of two vectors x, y where y_i = P(data > x_i)'''
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    y = np.arange(n, 0, -1) / n
    x = sorted_data
    
    return x, y

def keep_positive(data_list):
    '''Takes a LIST of tuples (x, y), and filters out 
    negative and zero entries (in both x and y) in each tuple.'''
    cleaned_data = []
    for x, y in data_list:
        mask = (x > 0) & (y > 0)
        cleaned_data.append((x[mask], y[mask]))
    return cleaned_data
 
def non_outliers(x, m):
    '''Takes an array x of data and an integer m,
    and returns a boolean array indicating whether each 
    value is within m standard deviations of the mean.'''
    mu = np.mean(x)
    sigma = np.std(x)
    return np.abs(x - mu) < m * sigma
    
def reject_outliers(data, m=2):
    '''Takes a tuple (x, y) where x and y are log-transformed arrays
    and removes outliers using m-sigma filtering.'''
    x, y = data
    
    mask_x = non_outliers(x, m)
    mask_y = non_outliers(y, m)
    mask = mask_x & mask_y
    return (x[mask], y[mask])
    
def linear_regression(X, y):
    '''Fits a linear model y = mX + b and returns (m, b, r^2).'''
    
    model = linear_model.LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, y)
    
    m = model.coef_[0]
    b = model.intercept_
    r2 = model.score(X_reshaped, y)
    
    return m, b, r2

def make_graphs_d(data, title, labels, ylabel='', xlabel='', filename_suffix=''):
    """Create log-log plots with best-fit lines for each distribution."""
    for i, ((X, y), label) in enumerate(zip(data, labels)):
        
        m, b, r2 = linear_regression(X, y)
        
        plt.figure()
        plt.scatter(X, y, label=label, s=5, alpha=0.5)
        plt.plot(X, b + m * X, 'r-', label=f'y = {m:.2f}x + {b:.2f}, $R^2$ = {r2:.3f}')
        
        plt.title(f"{title}: {label}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        
        if not os.path.exists('../latex/figs'):
            os.makedirs('../latex/figs')
        
        fname = f"plot_1.1_d_{filename_suffix}_{label.replace(' ', '_')}.png"
        plt.savefig(f"../latex/figs/{fname}")
        plt.show()

# %%
Xi = [normal_draws, weibull_draws, pareto_draws]
names = ["Normal", "Weibull", "Pareto"]

data_pdf = [pdf(Xi[i], bins=50) for i in range(3)]
data_pdf = keep_positive(data_pdf)

data_pdf_log = [(np.log(X), np.log(y)) for (X, y) in data_pdf]
data_pdf_clean = [reject_outliers(d, m=3) for d in data_pdf_log]

make_graphs_d(data_pdf_clean, 'Frequency plot (PDF, log-log)', names, 
              ylabel='log(Frequency)', xlabel='log(Value)', filename_suffix='frequency')

# %%
data_ccdf = [ccdf(Xi[i]) for i in range(3)]
data_ccdf = keep_positive(data_ccdf)

data_ccdf_log = [(np.log(X), np.log(y)) for (X, y) in data_ccdf]
data_ccdf_clean = [reject_outliers(d, m=3) for d in data_ccdf_log]

make_graphs_d(data_ccdf_clean, 'Rank plot (CCDF, log-log)', names,
              ylabel='log(P(X>x))', xlabel='log(Value)', filename_suffix='rank')



# %%
