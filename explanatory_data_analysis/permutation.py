#permutation sampling is a great way to simulate
# the hypothesis that two variables have identical probability distributions.
# e.g to test if two subsample comes from the same population 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def permutation_sample(data1,data2):
    data=np.concatenate((data1,data2))

    permutated_data=np.random.permutation(data)

    sample1=permutated_data[:len(data1)]
    sample2=permutated_data[len(data1):]

    return sample1,sample2

def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    # difference on smaple means
    zs = np.concatenate([xs, ys])

    list=np.empty(nmc)
    for j in range(999):
        np.random.shuffle(zs) # shuffle for 999 tinmes

        list[j]=np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return list

xs = np.array([24,43,58,67,61,44,67,49,59,52,62,50])
ys = np.array([42,43,65,26,33,41,19,54,42,20,17,60,37,42,55,28])
list_a=exact_mc_perm_test(xs, ys, 999)
print(list_a)


sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
sns.distplot(list_a,color="r",bins=30,kde=True) #kde=true，显示拟合曲线
plt.title('Permutation Test')
plt.xlabel('difference')
plt.ylabel('distribution')
plt.show()

















import numpy as np

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


'''
for i in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june,rain_november)


    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()



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
'''

import pandas as pd
import numpy as np
import ssl
import seaborn as sns
import matplotlib.pyplot as plt

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

print(df.head())
print(df['ID'].unique)

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

