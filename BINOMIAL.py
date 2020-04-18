# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults=np.random.binomial(100,0.05,size=10000)

# Compute CDF: x, y
x,y=ecdf(n_defaults)


# Plot the CDF with axis labels
plt.plot(x,y,marker='.',linestyle='none')
plt.xlabel('the number of defaults')
plt.ylabel('cdf')



# Show the plot
plt.show()

# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
plt.hist(n_defaults,bins=bins,normed=True)

# Label axes
plt.xlabel('n_defeults')
plt.ylabel('')


# Show the plot
plt.show()





















