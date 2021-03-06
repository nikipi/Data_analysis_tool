import pandas as pd
import numpy as np
import ssl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

ssl._create_default_https_context = ssl._create_unverified_context
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

url = r'https://assets.datacamp.com/production/repositories/470/datasets/df6e0479c0f292ce9d2b951385f64df8e2a8e6ac/frog_tongue.csv'

df = pd.read_csv(url, sep=',',header=0,engine='python',comment='#')
df = df.rename(columns=lambda x: x.replace("'","").replace('"','').replace(" ",""))
'''
print(df.head())
print(df['ID'].unique)
print(df.shape[0])

print(df.dtypes)
print(df.columns)

# Make bee swarm plot
_ = sns.swarmplot(x='ID',y='impactforce(mN)',data=df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()


print(df.groupby('ID').mean())
'''
pd.set_option('display.float_format',lambda x : '%.4f' % x)
force_a=df[df['ID']=='I']
force_a=force_a['impactforce(mN)']
print(force_a.mean(axis=0))

pd.set_option('display.float_format',lambda x : '%.4f' % x)
force_b=df[df['ID']=='II']
force_b=force_b['impactforce(mN)']
print(force_b.mean(axis=0))


def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1)-np.mean(data_2)

    return diff


#################################
#置换检验  permutation  test #####
#################################

#permutation test on frog data
# null hypothesis is the distribution of strike force for the two frog are identical
# the test difference=the observed difference
#p = np.sum(he test difference >=the observed difference) / len(test)

#often use permutation when the sample is small

#根据原有SZIE打乱原有数据
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

#用FUNC多次处理打乱的数据
def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1,perm_sample_2)

    return perm_replicates

#the hypothesis that the  of strike forces for the two frogs are identical

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a,force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a,force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >=empirical_diff_means) / len(perm_replicates)

#########################################################################
#                p-value
# a test statistic equally or more extreme than the one you observed ####
# given that the null hypothesis is true          #######################
#########################################################################

# Print the result
print('p-value =', p)
# p is small reject the null hypo, forces for the two frogs are not identical


##############################
# bootstrap hypo test  #######
##############################

# get bootstrap then process func
def bootstrap_replicate_1d(data,func):
    """Generate bootstrap replicate of 1D data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)
# repeat bootstrap
def draw_bs_reps(data,func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)

    return bs_replicates
# A one-sample bootstrap hypothesis test
# only know the mean force of  frog c is 550
# don't have the original data, cannot do a permutation test
# null hypothesis the mean force of frog b and c is equal
#p = np.sum (bs_replicates >= np.mean(force_b)) / 10000

# Make an array of translated impact forces: translated_force_b
force_c = force_b-np.mean(force_b)+550

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(force_c,np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates >= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)


###############################################################
#Hypothesis that Frog A and Frog B have the same mean force  ##
# but not necessarily the same distribution ###################
################################################################



forces_concat=pd.concat([force_a,force_b])
# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

#shift both arrays to have the same mean,
# since we are simulating the hypothesis that their means are, in fact, equal.
# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a-bs_replicates_b

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means)/ 10000
print('p-value =', p)



###################


# Compute the difference in mean sperm count: diff_means
diff_means = np.mean(control)-np.mean(treated)

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control,treated)))

# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control-bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)


# 153 House Democrats and 136 Republicans voted yea. However, 91 Democrats and 35 Republicans voted nay.
# Did party affiliation make a difference in the vote?


#the probability of observing a test statistic equally or more extreme than
#the one you observed, given that the null hypothesis is true.

##############################################
# The vote for the Civil Rights Act in 1964 ####
##############################################

#the null hypothesis is the fraction of yea vote has no difference between two parties
#p = np.sum(perm_replicates <= 153/244(the observed fraction)) / len(perm_replicates)


import numpy as np

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)


def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1,perm_sample_2)
    return perm_replicates

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac =np.sum(dems)/ len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems,reps,frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)


##########################################
#Hypothesis test on Pearson correlation ##
##########################################


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat=np.corrcoef(x,y)


    # Return entry [0,1]
    return corr_mat[0,1]

r_obs = pearson_r(illiteracy,fertility)

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted,fertility)

# Compute p-value: p
p = np.sum(perm_replicates>=r_obs)/10000
print('p-val =', p)
