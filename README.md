# PCA
Principal Component Analysis from scratch on jaffedbase Dataset

The dataset contains face images of 30 different individuals. Each image is has a RGB format with the size of 128*128. You can convert the images into gray scale. To speed up your code, you can resize the images into 64*64.

Eigenfaces are sets of eigenvectors which can be used to work with face recognition applications. Each eigenface, as we'll see in a bit, appears as an array vector of pixel intensities. We can use PCA to determine which eigenfaces contribute the largest amount of variance in our data, and to eliminate those that don't contribute much. This process lets us determine how many dimensions are necessary to recognize a face as 'familiar'
