from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.sans-serif']=['SimHei']
x=[10,20,30,40,50,60,70,80,90,100]
# x = [55,60,65,70,75,80,85,90,95,100]
# x_tick=[50,55,60,65,70,75,80,85,90,95,100] #不仅xlim要改变，x_tick也要改变才行
# figsize=(12.8,8)
figsize=(5,3) #PR size


plt.figure(figsize=figsize,dpi=1200)

der_y = [11.2 * i for i in range(1,11)]
der_std = [0 for i in range(1,11)]

ours_y = [1.12 * 0.01 for i in range(1,11)]
ours_std = [0 for i in range(1,11)]

more_y = [24.11 + i * 0.13 for i in range(1,11)]
more_std = [0 for i in range(1,11)]

l_der = plt.errorbar(x, der_y, yerr=der_std,fmt='o-',color='green')

l_ours = plt.errorbar(x, ours_y, yerr = ours_std,fmt='o-',color='red')

l_more = plt.errorbar(x, more_y, yerr = more_std,fmt='o-',color='blue')



# plt.legend(handles=[ joint_handle,l_ours,l_der,l_icarl, l_more, l_wa, l_darker,l_ucir,l_podnet],\
#            labels=[ "Joint", 'Ours','DER', 'ICaRL', "MORE",'WA',"Darker++","UCIR","PODNet"], loc='best',fontsize=20)
# plt.legend(handles=[ joint_handle,l_ours,l_der,l_icarl, l_more, l_wa, l_darker,l_ucir,l_podnet],\
#            labels=[ "Joint", 'Ours','DER', 'ICaRL', "MORE",'WA',"Darker++","UCIR","PODNet"], loc='best',fontsize=20)
# plt.legend(handles=[ l_der, l_more, l_ours],\
#            labels=[ 'DynaER',"MORE",'Ours'], loc='best',fontsize=10)
# plt.legend(handles=[l_more],\
#            labels=["MORE"], loc='best',fontsize=10)
# y_ticks = np.arange(30, 101, 10)
y_ticks = np.arange(0, 111, 10)

plt.grid(ls="--",c='gray',axis='y',linewidth=1)
plt.xlim((9, 101))

plt.ylim((9, 115))
plt.xticks(x,fontsize=10)
plt.yticks(y_ticks,fontsize=10)


# plt.title('Base 10 Increment 10')

plt.xlabel('Number of Classes',fontsize=10)
# plt.xlabel('任务数量',fontsize=35)
# plt.xlabel('参数量',fontsize=35)

plt.ylabel('Trainable Parameters(Millions)',fontsize=10)
# plt.ylabel('Trainable Parameters(Millions)',fontsize=10)

plt.savefig('./cifar100_b10i10_model_params.png',bbox_inches='tight')
# plt.show()
