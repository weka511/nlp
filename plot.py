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

from matplotlib.pyplot   import figure, legend, plot, savefig, show, title, xlabel, ylabel
from torch import load

if __name__=='__main__':
    parser = ArgumentParser('Plot training from word2vector')
    parser.add_argument('input',       nargs='+',                                           help = 'Files to process')
    parser.add_argument('--output',                  default = 'plot',                      help = 'Output file name')
    parser.add_argument('--show',                    default = False, action='store_true', help ='Show plots')
    args = parser.parse_args()
    figure(figsize=(10,10))
    corpus   = None
    embedded = None
    for i,file_name in enumerate(args.input):
        loaded      = load(file_name)
        loaded_args = loaded['args']
        print (loaded.keys())
        print (loaded_args)

        if i==0:
            corpus = loaded_args.corpus
            embedded = loaded_args.m #FIXME
        else:
            pass #FIXME
        Epochs = [1,2,3] #FIXME
        Losses = [1,2,3] #FIXME
        plot(Epochs,Losses,label=f'{file_name}')

    title(f'{corpus} -- Embedding dimensions={embedded}')
    xlabel('Epoch')
    ylabel('Loss')
    legend()
    savefig(args.output)
    if args.show:
        show()
