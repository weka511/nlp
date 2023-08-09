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

'''
    This module is responsibe for reading data from corpus.
    Data is supported in several different formats, e.g. text, zipped XML.
'''

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from os import remove
from os.path import exists, join
from pathlib import Path
from re import split
from sys import exc_info, stderr
from time import time
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
        '''Iterate through the words in the sentence'''
        class SentenceIterator:
            def __init__(self,words):
                self.words = words
                self.n = 0
            def __next__(self):
                if self.n < len(self.words):
                    self.n += 1
                    return self.words[self.n-1]
                else:
                    raise StopIteration

        return SentenceIterator(self.words)

    def __len__(self):
        '''Length of sentence in words'''
        return len(self.words)

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
    A Corpus comprising one or more text files
    '''
    def __init__(self,dataset):
        self.dataset = dataset

    def generate_tags(self,max_files=None,log_file=stderr):
        with open(self.dataset,encoding='utf-8') as text_file:
            for tag in pos_tag( word_tokenize( text_file.read())):
                yield tag


class CorpusZippedXml(Corpus):
    '''
    A corpus comprising a set of zipped XML files. I'm parsing the xml myself, as the blogs.zip
    dataset contains numersous encoding errors.
    '''
    def __init__(self,dataset):
        self.dataset = dataset

    def generate_tags(self,max_files=None,log_file=stderr,encoding='ISO-8859-1'):
        '''
        Convert text from corpus to a list of words and tags
        '''
        with zf.ZipFile(self.dataset) as zipfile:
            n_files = 0
            if max_files == None:
                max_files = len(zipfile.namelist())
            for file_name in zipfile.namelist():
                if file_name.endswith('/'): continue
                if n_files > max_files: return
                contents = zf.Path(zipfile, at=file_name).read_text(encoding=encoding)
                start = 0
                while True:
                    p1 = contents.find('<post>',start)
                    if p1 < 0: break
                    p1 += len('<post>')
                    p2 = contents.find('</post>',p1)
                    start = p2 + len('</post>')
                    p2 -= 1
                    for tag in pos_tag( word_tokenize( contents[p1:p2])):
                        yield tag
                n_files += 1



def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='./data', help='Path to data files (default: %(default)s)')
    parser.add_argument('--dataset', default='...', help='Name of dataset (default: %(default)s)')
    parser.add_argument('--n', default=10,type=int, help='Number of files (for testing)')
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
