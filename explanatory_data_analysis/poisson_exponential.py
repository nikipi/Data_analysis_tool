# Draw 10,000 samples out of Poisson distribution: samples_poisson
import numpy as np
samples_poisson=np.random.poisson(10,size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i],p[i],size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))



def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=1)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=1)

    return t1 + t2



# Draw samples of waiting times: waiting_times
waiting_times=successive_poisson(764,715,size=100000)

# Make the histogram
plt.hist(waiting_times,bins=100,normed=True,histtype='step')


# Label axes
plt.xlabel('')
plt.ylabel('')

# Show the plot
plt.show()


















