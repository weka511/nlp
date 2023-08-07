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

'''Read data from corpus'''

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from os import remove
from os.path import exists, join
from pathlib import Path
from sys import exc_info, stderr
from time import time
from xml.dom import minidom
from xml.parsers.expat import ExpatError
import zipfile as zf
from nltk import word_tokenize, pos_tag

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

class Sentence:
    '''
    Represent a sentence as a list of lower case words without punctuation
    '''
    def __init__(self):
        self.words = []

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return len(self.words)

    def __next__(self):
        if self.pos < len(self.words):
            self.pos += 1
            return self.words[self.pos-1]
        else:
            raise StopIteration

    def __str__(self):
        return ' '.join(self.words)

    def add(self,word):
        self.words.append(word.lower())

    def clear(self):
        self.words.clear()

class Corpus(ABC):
    '''
    Represents a text corpus
    '''
    @staticmethod
    def create(dataset,format='ZippedXml'):
        match (format):
            case 'Text' :
                return CorpusText(dataset)
            case 'ZippedXml':
                return CorpusZippedXml(dataset)


    @abstractmethod
    def generate_tags(self):
        '''
        Convert text from corpus to a list of words and tags
        '''
        ...

    def generate_sentences(self,n=None):
        sentence = Sentence()
        for word,tag in self.generate_tags(n):
            match tag:
                case '.' |':' | ';' :
                    yield sentence
                    sentence.clear()
                case ',' |  '"' | "'" | '`':
                    pass
                case _:
                    sentence.add(word)
        if len(sentence)>0:
            yield sentence


class CorpusText(Corpus):
    '''
    A Corpus comprining one or more text files
    '''
    def __init__(self,dataset):
        self.dataset = dataset

    def generate_tags(self,max_files=None,log_file=stderr):
        with open(self.dataset,encoding='utf-8') as text_file:
            for tag in pos_tag( word_tokenize( text_file.read())):
                yield tag


class CorpusZippedXml(Corpus):
    '''
    A corpus comprising a set of zipped XML files
    '''
    def __init__(self,dataset):
        self.dataset = dataset

    def generate_tags(self,max_files=None,log_file=stderr):
        '''
        Convert text from corpus to a list of words and tags
        '''
        with zf.ZipFile(self.dataset) as zipfile:
            n_files = 0
            for file_name in zipfile.namelist():
                if file_name.endswith('/'): continue
                n_files += 1
                if max_files != None and n_files > max_files: return
                path = zf.Path(zipfile, at=file_name)
                try:
                    contents = path.read_text(encoding='ISO-8859-1')
                    doc = minidom.parseString(contents)
                    for post in doc.getElementsByTagName('post'):
                        for tag in pos_tag( word_tokenize( post.firstChild.nodeValue)):
                            yield tag
                except UnicodeDecodeError as err:
                    log_file.write(f'UnicodeDecodeError: {file_name}  {err.lineno} {err}\n')
                except ExpatError as err:
                    exc_type, exc_obj, exc_tb = exc_info()
                    log_file.write(f'ExpatError: {file_name} {exc_tb.tb_lineno} {err.lineno} {err}\n')


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='./data', help='Path to data files (default: %(default)s)')
    parser.add_argument('--figs', default='./figs', help='Path to plot files (default: %(default)s)')
    parser.add_argument('--plot', default = Path(__file__).stem)
    parser.add_argument('--save', default = Path(__file__).stem)
    parser.add_argument('--dataset', default='...', help='Name of dataset (default: %(default)s)')
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('--N', default=100,type=int, help='Number of iterations (default: %(default)s)')
    parser.add_argument('--n', default=10,type=int)
    parser.add_argument('--freq', default=5,type=int, help='Controls how frequently progress will be shown (default: %(default)s)')
    parser.add_argument('--format', choices=['ZippedXml', 'Text'], default='ZippedXml', help='Format for corpus (default: %(default))')

    return parser.parse_args()

if __name__=='__main__':
    start  = time()
    args = parse_args()
    corpus = Corpus.create(join(args.data,args.dataset),format=args.format)

    for word,tag in corpus.generate_tags(args.n):
        print (word,tag)

    for s in corpus.generate_sentences(args.n):
        print (s)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
