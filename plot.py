#    Copyright (C) 2021 Simon A. Crase
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
#
#

from argparse            import ArgumentParser
from glob                import glob
from matplotlib.pyplot   import figure, legend, plot, savefig, show, suptitle, title, xlabel, ylabel
from torch import load

if __name__=='__main__':
    parser = ArgumentParser('Plot training from word2vector')
    parser.add_argument('input',    nargs='+',                                        help = 'Files to process (may be globbed)')
    parser.add_argument('--output',            default = 'plot',                      help = 'Output file name')
    parser.add_argument('--show',              default = False, action='store_true',  help ='Show plots')
    parser.add_argument('--chain',             default = False, action='store_true',  help ='Chain plots along x axis')
    args = parser.parse_args()
    figure(figsize=(10,10))
    corpus     = None
    embeddings = None
    window     = None
    T          = 0    # offset for plots of chain specified

    for i,file_name in enumerate([name for file_spec in args.input for name in glob(f'{file_spec}')]):
        loaded      = load(file_name)
        loaded_args = loaded['args']

        if i==0:
            corpus    = loaded_args.corpus
            embedding = loaded_args.embedding
            window    = loaded_args.window
        else:
            if corpus != loaded_args.corpus    or \
            embedding != loaded_args.embedding or \
            window    != loaded_args.window:
                print ('There is an inconsistency between')
                print (f'{corpus}: embedding={embedding}, window {window} and')
                print (f'{corpus}: embedding={loaded_args.embedding}, window {loaded_args.window} and')

        plot([t+T for t in loaded['Epochs']],loaded['Losses'],
             label=f'{file_name}, learning rate={loaded_args.lr}, momentum={loaded_args.alpha}')

        if args.chain:
            T += len(loaded['Epochs'])

    suptitle(f'{corpus}')
    title(f'Embedding = {embedding}, window = {loaded_args.window}')
    xlabel('Epoch')
    ylabel('Loss')
    legend()
    savefig(args.output)

    if args.show:
        show()
