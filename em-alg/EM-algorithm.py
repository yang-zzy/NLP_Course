import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.stats import norm

height_list = pd.read_csv('./height_data.csv', header=0, names=['height'])
height_list = [row['height'] for index, row in height_list.iterrows()]
N = len(height_list)

pai = 0.5
uF = 160
sF = 1
uM = 180
sM = 1
gama = np.zeros(N)
tol = 1e-6
max_iteration = 200
pbar = tqdm.trange(max_iteration)

for j in enumerate(pbar):
    # E-STEP
    for i in range(N):
        gama[i] = pai * norm.pdf(height_list[i], uF, sF) / (
                pai * norm.pdf(height_list[i], uF, sF) + (1 - pai) * norm.pdf(height_list[i], uM, sM))
    # M-STEP
    gama_total = uF_total = sF_total = uM_total = sM_total = 0
    for i in range(N):
        gama_total += gama[i]
        uF_total += height_list[i] * gama[i]
        uM_total += height_list[i] * (1 - gama[i])
    pai_new = gama_total / N
    uF_new = uF_total / gama_total
    uM_new = uM_total / (N - gama_total)
    for i in range(N):
        sF_total += gama[i] * np.power(height_list[i] - uF_new, 2)
        sM_total += (1 - gama[i]) * np.power(height_list[i] - uM_new, 2)
    sF_new = np.sqrt(sF_total / gama_total)
    sM_new = np.sqrt(sM_total / (N - gama_total))
    if abs(pai - pai_new) < tol and abs(uF - uF_new) < tol and abs(uM - uM_new) < tol and abs(
            sF - sF_new) < tol and abs(sM - sM_new) < tol:
        print('EM算法已经收敛！')
        break
    pai = pai_new
    uF = uF_new
    uM = uM_new
    sF = sF_new
    sM = sM_new
    log_likelihood = np.dot(np.log(pai * norm.pdf(height_list, uF, sF) + (1 - pai) * norm.pdf(height_list, uM, sM)),
                            gama)
    pbar.set_postfix(pai=pai, uF=uF, uM=uM, sF=sF, sM=sM, log_like=log_likelihood)
    time.sleep(0.2)

log_likelihood = np.dot(np.log(pai * norm.pdf(height_list, uF, sF) + (1 - pai) * norm.pdf(height_list, uM, sM)), gama)
print('女生占比:{:.4%},\n女生身高均值:{:.2f}\n女生身高标准差:{:.2f}\n男生身高均值:{:.2f}\n男生身高标准差:{:.2f}'.format(
    pai, uF, sF, uM, sM))
print('log_likelihood : ', log_likelihood)

x = np.linspace(min(height_list), max(height_list), 200)


def y(x):
    y = pai * norm.pdf(x, uF, sF) + (1 - pai) * norm.pdf(x, uM, sM)
    return y * N


plt.hist(height_list, bins=20, color="w", label="直方分布图", edgecolor='k')
plt.xlabel('height/cm')
plt.ylabel('Count')
plt.plot(x, y(x), 'r-', linewidth=1, label='f(x)')
plt.show()
