#!/usr/bin/env python

#    Copyright (C) 2021-2023 Simon A. Crase   simon@greenweaves.nz
#
#    This is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This software is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>

'''Explore variability of Bradley-Terry'''

from cycler            import cycler
from matplotlib.pyplot import figure,bar,xlabel,ylabel,legend,rc, plot,savefig, title, scatter, show
from numpy             import argsort, exp, zeros, int32, sum, sqrt, log, argmin,mean,std
from numpy.random      import rand
from os                import walk
from os.path           import join
from pandas            import read_csv
from random            import random, randrange, gauss,sample
from time              import time


rc('lines', linewidth=2)
rc('axes',
   prop_cycle=(cycler(color=['c', 'm', 'y','b','c', 'm', 'y','b']) +
                  cycler(linestyle=[ '--', '--','--','--',':', ':', ':',':'])))

N_TRIALS       = 25
N_CONTESTS     = 20
N_MAX          = 50#200#None
MAX_ITERATIONS = 1000
FREQUENCY      = 150
PLOT_FILE      = 'bt-iterations'
EPSILON        = 1e-6

train_data    = None
df_colours    = None
xkcd_colours  = None
for dirname, _, filenames in walk('./'):
    for filename in filenames:
        path_name = join(dirname, filename)
        if filename.startswith('train'):
            train_data = read_csv(path_name)
        if filename.startswith('colors'):
            df_colours = read_csv(path_name)
            xkcd_colours = df_colours.XKCD_COLORS.dropna()

# create_wins_losses
#
# Compute a matrix w[i,k] -- the number of wins for i competing with k
def create_wins_losses(Lambdas):
    w  = zeros((N_MAX,N_MAX))
    for i in range(N_MAX):
        for j in range(N_CONTESTS):
            k = (i+randrange(1,N_MAX)) % N_MAX
            if random() < Lambdas[i]/(Lambdas[i] + Lambdas[k]):
                w[i,k] += 1
            else:
                w[k,i] += 1
    return w

# update
#
# Update probablities

def update(p,w_symmetric,W,N):
    p1 = zeros(N)
    for i in range(N):
        Divisor = 0
        for j in range(N):
            if i!=j and p[i]+p[j]>0:
                Divisor += w_symmetric[i,j]/(p[i]+p[j])
        p1[i] = W[i]/Divisor
    return p1/sum(p1)

# normalize
#
# Make elements of a vector sum to 1 so it can be used as a probability

def normalize(p):
    return p / sum(p)

def rmse(predictions, targets):
    return sqrt(((predictions - targets) ** 2).mean())


start          = time()
Targets        = train_data.target.to_numpy()
if N_MAX<len(Targets):
    Targets = sample(list(Targets),N_MAX)
Betas          = sorted(Targets)
index_min_beta = argmin([abs(b) for b in Betas])

Lambdas        = exp(Betas)
N,_            = train_data.shape

fig            = figure(figsize=(10,10))
# axes           = fig.subplots(nrows=1,ncols=1)

# plot(range(N_MAX),Betas,#Lambdas/sum(Lambdas),
             # label     = r'$\lambda$',
             # linestyle = '-', linewidth=6, color='k')

Scores = zeros((N_MAX,N_TRIALS))

for trial in range(N_TRIALS):
    w              = create_wins_losses( Lambdas)
    w_symmetric    = w + w.transpose()
    W              = sum(w,axis=1)  # Number won by i
    Ps             = normalize(rand(N_MAX))

    for k in range(MAX_ITERATIONS):
        p1 = update(Ps,w_symmetric,W,N_MAX)
        if (abs(Ps-p1)<EPSILON*p1).all():
            Ps = p1
            break
        Ps  = p1
    Ls = log(Ps)
    offset = Ls[index_min_beta]-Betas[index_min_beta]
    LLs = [l - offset for l in Ls]
    Scores[:,trial] = LLs
    # plot(range(N_MAX),LLs,  linestyle = ':',color=xkcd_colours[trial%len(xkcd_colours)])
    if trial%10==0:
        print (f'Trial {trial}')

mu = mean(Scores,axis=1)
sigma = std(Scores,axis=1)
plot (mu)
plot (sigma)
legend()
xlabel('index')
ylabel('p')
elapsed = int(time() - start)
title(f'{N_MAX} Contestants, {N_CONTESTS} contests. Time = {elapsed} seconds, eps={EPSILON}, k={k}')

    # axes[1].scatter(Lambdas/sum(Lambdas),p,s=5,label=r'$\lambda$ vs p',color='b')
    # axes[1].plot(Lambdas/sum(Lambdas),Lambdas/sum(Lambdas),label='Ideal',color='k')
    # axes[1].set_xlabel(r'$\lambda$')
    # axes[1].set_ylabel('p')
    # axes[1].legend()
    # axes[1].set_title(f'RMS (log domain)= {rmse(log(p), log(Lambdas/sum(Lambdas))):.2f}')
fig.savefig(f'{PLOT_FILE}')

show()
