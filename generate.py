#! /usr/bin/python2

from random import random

if __name__ == '__main__':
    outfile = open('genre.init', 'w')
    layer_sizes = [13, 50, 4]

    s = ''
    for size in layer_sizes:
        s += str(size) + ' '
    outfile.write(s[:-1] + '\n')
    print(s[:-1])

    for (i, layer_size) in enumerate(layer_sizes[1:]):
        for j in range(layer_size):
            s = ''
            for k in range(layer_sizes[i]):
                s += str(random()) + ' '
            outfile.write(s[:-1] + '\n')
            print(s[:-1])
