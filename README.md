# Phase congruency #

An implementation of phase congruency image features detection: edges and corners.

Kovesi, P.D.: Image features from phase congruency. Videre: Journal of Computer Vision Research 1(1999)
http://mitpress.mit.edu/e-journals/Videre/

![Example image](/example/1.png)

![Example of edges detection](/example/edges.png)

![Example of corners detection](/example/corners.png)

## To build: ##

### Windows ###

Microsoft Visual Studio required.

### Linux ###

    make

## To use: ##

    testPhase input_file_name.png output_file_name.png

for example:

    testPhase example/1.png example/out.png
