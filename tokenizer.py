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

def read_text(file_name='chapter1.txt'):
    with open(file_name) as text_file:
        for line in text_file:
            yield line.strip()

def extract_sentences(text):
    buffer = []
    for line in text:
        parts = line.split('.')
        buffer.append(parts[0])
        while len(parts)>1:
            parts = parts[1:]
            if len(parts[0])==0:
                buffer.append(parts[0])
                parts = parts[1:]
                yield ' '.join(buffer)
                buffer = []
                break
            if parts[0]=='"':
                buffer.append(parts[0])
                parts = parts[1:]
            yield ' '.join(buffer)
            buffer = []
            if len(parts)>0:
                buffer.append(parts[0])
        x=0

if __name__=='__main__':
    for sentence in extract_sentences(read_text()):
        print (sentence)
