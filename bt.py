from cycler            import cycler
from matplotlib.pyplot import figure,bar,xlabel,ylabel,legend,rc, plot,savefig, title, scatter, show
from numpy             import argsort, exp, zeros, int32, sum, sqrt, log
from numpy.random      import rand
from os                import walk
from os.path           import join
from pandas            import read_csv
from random            import random, randrange, gauss
from time              import time


rc('lines', linewidth=2)
rc('axes',
   prop_cycle=(cycler(color=['c', 'm', 'y','b','c', 'm', 'y','b']) +
                  cycler(linestyle=[ '--', '--','--','--',':', ':', ':',':'])))

N_TRIALS       = 1
N_CONTESTS     = 20
N_MAX          = 500#None
MAX_ITERATIONS = 1000
FREQUENCY      = 150
PLOT_FILE      = 'bt-iterations'
EPSILON        = 1e-8

train_data    = None

for dirname, _, filenames in walk('./'):
    for filename in filenames:
        path_name = join(dirname, filename)
        if filename.startswith('train'):
            train_data = read_csv(path_name)

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

for i in range(N_TRIALS):
    start          = time()
    Betas          = train_data.target.to_numpy()
    Lambdas        = exp(Betas)
    N,_            = train_data.shape
    if N_MAX==None:
        N_MAX = N
    else:
        Lambdas    = Lambdas[:N_MAX]
    w              = create_wins_losses( Lambdas)
    w_symmetric    = w + w.transpose()
    W              = sum(w,axis=1)  # Number won by i
    p              = normalize(rand(N_MAX))

    fig            = figure(figsize=(10,10))
    axes           = fig.subplots(nrows=2,ncols=1)
    axes[0].plot(range(N_MAX),Lambdas/sum(Lambdas),
                 label     = r'$\lambda$',
                 linestyle = '-',
                 color     = 'k')

    for k in range(MAX_ITERATIONS):
        p1 = update(p,w_symmetric,W,N_MAX)
        if (abs(p-p1)<EPSILON*p1).all():
            p = p1
            break
        p  = p1
        if k%FREQUENCY==0:
            axes[0].plot(range(N_MAX),p,
                         label = f'iteration={k}')

    axes[0].plot(range(N_MAX),p,
                 label     = f'Final',
        linestyle = '-',
        color     = 'r')
    axes[0].legend()
    axes[0].set_xlabel('index')
    axes[0].set_ylabel('p')
    elapsed = int(time() - start)
    axes[0].set_title(f'{N_MAX} Contestants, {N_CONTESTS} contests. Time = {elapsed} seconds, eps={EPSILON}, k={k}')

    axes[1].scatter(Lambdas/sum(Lambdas),p,s=5,label=r'$\lambda$ vs p',color='b')
    axes[1].plot(Lambdas/sum(Lambdas),Lambdas/sum(Lambdas),label='Ideal',color='k')
    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel('p')
    axes[1].legend()
    axes[1].set_title(f'RMS (log domain)= {rmse(log(p), log(Lambdas/sum(Lambdas))):.2f}')
    fig.savefig(f'{PLOT_FILE}-{i}')

show()
