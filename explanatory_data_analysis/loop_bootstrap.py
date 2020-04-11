
#当我们做一个统计计算时，经常会发现无法收集到全部的样本数据
#出于这种无奈，我们只好退而求其次地相信选取其中的某一段数据（1883年到2015年）
#经过反复的bootstrap replicate（相当于无限重复抽样的次数，
#以弥补我们在客观现实世界中无法发无限次重复实验的遗憾），得到一系列的样本
#从中认为这就是最接近客观世界的样本。

import numpy as np
def bootstrap(data, func):
    """Draw bootstrap replicates."""

   bs_sample=np.random.choice(data,len(data))

    return func(bs_sample)


# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates =np.empty(10000)

for i in range(10000):
    bs_replicates[i]=bootstrap(data,np.mean)


# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


# confidence interval
np.percentile(bs_replicates,[2.5,97.5])

