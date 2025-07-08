# # Project C

import matplotlib.pyplot as plt
import numpy as np

#
# ***
# ### Question C.1.
#

#1. Euler–Maruyama might produce negative stock market index $S_n$, which is not realistic. The exponential function in (C.2.2) avoids this fault.

#2. Equation (C.2.2) (which takes a exp function) applies the exact solution of the SDE,
# while The Euler one is only a first-order numerical approximation, neglecting the higher-order terms.
# So (C.2.2) grasps the dynamics better, yielding much more accurate results.

#3. (C.2.2) preserve the log-normal distribution of $S(t)$ but Euler-Maruyana method introduces more errors. 

#
# ***
# ### Question C.2.
#

def simulate_gbm(mu, P, T, N):
    sigma = 0.15
    S0 = 1.0
    dt = T/N
    S = np.zeros((P, N+1))
    S[:,0] = S0

    #Use the property of exponential function to generate the numerical scheme in a vectorized manner
    dWs = np.random.normal(0.0, np.sqrt(dt), size=(P, N))
    S[:, 1:] = S0 * np.exp( np.cumsum((mu - 0.5*sigma**2)*dt + sigma*dWs, axis=1) )

    return S

#
# ***
# ### Question C.3.
#

mu = 0.05
T = 40
N = 12*T
P = 100000

S = simulate_gbm(mu, P, T, N)

def paths_plotting(S, T, N, num_paths=5):
    t = np.linspace(0, T, N+1)

    percentile_5th = np.percentile(S, 5, axis=0)
    percentile_95th = np.percentile(S, 95, axis=0)
    mean = np.mean(S, axis=0)

    plt.figure(figsize=(12, 6))

    for i in range(num_paths):
        plt.plot(t, S[i], ':', label = f'{i+1}th Path')
        
    plt.plot(t, mean, color = 'b', label = 'Mean Path')
    plt.plot(t, percentile_5th, 'r--', label = '5th Percentile')
    plt.plot(t, percentile_95th, 'g--', label = '95th Percentile')

    plt.title('Stock Prices Simulation with model GBM')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price $S_(t)$')
    plt.legend()
    plt.grid(True)

    plt.show()

paths_plotting(S, T, N, num_paths=5)

#
# ***
# ### Question C.4.
#

def pension_value(paths, T, M):
    N = 12*T   #We need paths.shape[1] <= N actually
    H_T = M * np.sum(1.0 / paths[:,:N], axis=1)
    V_T = H_T * paths[:, N]
    return V_T

def verification_1():
    P = 3
    T = 2
    M = 1000
    N = 12 * T
    paths = np.ones((P, N + 1))
    V = pension_value(paths, T, M)
    
    print("Verification 1:")
    print("The expected result should be 1000/1*24/1 = 24000 (for 3 paths)")
    print(f"The actual result is V(T)={V}")

def verification_2():
    T = 1
    M = 1000
    N = 12*T
    paths = np.array([2**i for i in range(N+1)])
    paths = np.reshape(paths, (1, N+1)) #Turn a 1d array into a 1*(N+1) matrix

    V = pension_value(paths, T, M)

    expected = M* sum(1 / 2**i for i in range(12)) * 2**12
    
    print("Verification 2")
    print(f"The expected result should be {expected}")
    print(f"The actual result is V(T)={V}")

verification_1()
verification_2()

#
# ***
# ### Question C.5.
#

M = 1000
V = pension_value(S, T, M)

def histogram_plotting(V, bin_width = 50000):
    min_V, max_V = np.min(V), np.max(V)
    bin_edges = np.arange(
        np.floor(min_V/bin_width) * bin_width,
        np.ceil(max_V/bin_width) * bin_width + bin_width,
        bin_width
    )

    plt.figure(figsize=(10, 6))

    counts, bins, patches = plt.hist(V, bins=bin_edges, color='blue')

    #find the bin with the highest count
    max_idx = np.argmax(counts)

    plt.title('Distribution of the Final Pension Value')
    plt.xlabel('Final Value (GBP)')
    plt.ylabel('Number of Paths')
    plt.grid(True)
    plt.show()
    
    print(f'Bin with the highest count is [{bins[max_idx]:,.0f} GBP, {bins[max_idx + 1]:,.0f} GBP), with {counts[max_idx]:,.0f} paths.')

histogram_plotting(V)

#
# ***
# ### Question C.6.
#

#Suppose we invest for T years, and the monthly contribution is M, then the final value is:
#$$ V(T)=(\sum^{12T-1}_{n=0}\frac{M}{S(n)}) \cdot S(12T) $$
#As a GBM model, S(n) is proportional to S(0).
# So if we multiply $S(0)$ by a constant $C$, then $S(n)$ will be multiplied by $C$ as well.
#From the above equation, we can see that the constant $C$ will cancel out.
#Hence the final value is independent of S(0).

#Additionally, this is also a reason why we prefer the numerical method based on analytical solution instead of the Euler-Maruyama method.

#
# ***
# ### Question C.7.
#

amount_invested = M * N

p_loss = np.mean(V < amount_invested)
p_double = np.mean(V >= 2 * amount_invested)
p_2m = np.mean(V >= 2000000)

#Probability with 4 decimal places
print(f"Probability of making a loss: {p_loss:.4f}")
print(f"Probability of doubling the investment: {p_double:.4f}")
print(f"Probability of the investment being at least 2,000,000 GBP: {p_2m:.4f}")

#
# ***
# ### Question C.8.
#

def investment_analysis(T, mu_list, M_info, Vmin=1000000, P=100000):
    N = 12 * T
    
    M_array = np.arange(*M_info)  #M_info gives (start, end, step), where end is exclusive

    print(f"Invest analysis for T = {T}, M: {M_info}, mu = {mu_list}:")

    plt.figure(figsize=(10, 6))

    for mu in mu_list:
        S = simulate_gbm(mu, P, T, N)

        #find the probability of comfortable retirement
        #this is a vectorized method
        pension_factor = pension_value(S, T, 1)
        V_matrix = M_array[:, None] * pension_factor[None, :]
        R = np.mean(V_matrix >= Vmin, axis=1)

        plt.plot(M_array, R, label = f'mu = { int(mu*100) }%')

        #find what M is necessary to reach 95%
        #the condition R[0]<0.95 takes the case when only the first one reaches 95% into account, though it normally won't happen
        idx = np.argmax(R >= 0.95)
        if idx == 0 and R[0] < 0.95:
            print(f"μ = {int(mu*100)}%: Cannot reach 95% success within M ≤ {M_info[1]}")
        else:
            print(f"μ = {int(mu*100)}%: Minimum monthly M = £{M_array[idx]} is able to reach ≥95% probability")

    plt.axhline(0.95, linestyle='--', label='95% Threshold')
    plt.xlabel('Monthly Investment (GBP)')
    plt.ylabel('Probability of Comfortable Retirement R(M)')
    plt.title(f'Comfortable Retirement Probability against Monthly Investment (T = {T} yrs)')
    plt.grid(True)
    plt.legend()
    plt.show()

investment_analysis(40, [0.03, 0.05, 0.07], (0, 4001, 25))

#
# ***
# ### Question C.9.
#

investment_analysis(20, [0.03, 0.05, 0.07], (0, 8001, 25))

#Comments:

# At mu=3%: M increase from ~3300gbp to ~6300gbp with a approximately 90% increase.

# At mu=5%: M increase from ~2250gbp to ~5200gbp with a approximately 130% increase.

# At mu=7%: M increase from ~1500gbp to ~4250gbp with a approximately 180% increase.

# Generally, the required monthly investment M increases by around 70% to 200% when compared to the 40yr case.

# Graphically there is a significant right-shift of success probability against M.

#This phenomenon can be explained by:
# (i)Long-Term investment smoothes out the volatility of the stock market.
# (ii)The compounding property of the stock market is reduced.