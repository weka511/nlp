#!/usr/bin/env python

#   Copyright (C) 2023 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''td-idf algorithm'''

from argparse import ArgumentParser
from collections import ChainMap, Counter
from pathlib import Path
from time import time

import numpy as np
from spacy import load

def TfIdf(docnames=[]):
    nlp = load("en_core_web_sm")
    docnames = docnames
    dicts = ChainMap()
    word_counts = []
    for file_name in docnames:
        doc = nlp(Path(file_name).read_text(encoding="utf-8"))
        words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and not token.is_punct]
        word_counts.append(Counter(words))
        dicts.maps.append(word_counts[-1])
    tf = np.zeros((len(list(dicts)),len(docnames)))
    df = np.zeros((len(list(dicts))))
    for i,word in enumerate(list(dicts)):
        for j,wc in enumerate(word_counts):
            tf[i,j] = wc[word]
            if wc[word]>0:
                df[i] += 1
                z=0



if __name__=='__main__':
    start  = time()
    parser = ArgumentParser(__doc__)

    args = parser.parse_args()
    TfIdf(docnames=['gatsby.txt','erewhon.txt'])

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
