# imports
import numpy as np
import glob
from numpy import linalg as la
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


# read Images


def gridDisplay(image_list, title):
    fig1, axes_array = plt.subplots(17, 10)
    fig1.suptitle(title)
    fig1.set_size_inches(17, 10)
    k = 0
    for row in range(17):
        for col in range(10):
            im = np.array(Image.fromarray(image_list[k]).resize((64, 64)))
            image_plot = axes_array[row][col].imshow(im, cmap='gray')
            axes_array[row][col].axis('off')
            k = k + 1
    plt.show()


image = []
flattened_images = []
# reade images
for filename in glob.glob('jaffe/*.tiff'):
    im = Image.open(filename)
    im = np.asarray(im.resize((64, 64)))
    im = np.asarray(im, dtype=float) / 255.0
    image.append(im)
# show org images
gridDisplay(image, 'Original Images')

for i in range(len(image)):
    u = image[i].flatten()
    flattened_images.append(u)

A_transpose = np.matrix(flattened_images)
A = np.transpose(A_transpose)

mean = np.mean(A, 1)
b = mean.reshape(64, 64)
resized_mean = np.array(Image.fromarray(b).resize((64, 64)))
plt.imshow(resized_mean, cmap='gray')
plt.axis('off')
plt.title('Mean Face')
plt.show()

zero_mean = []
column = 0
Zero_mean_matrix = np.ones((4096, 170))

for values in flattened_images:
    # zm = values-mean
    zm = A[:, column] - mean
    # print("z",zm.shape)
    zm = np.squeeze(zm)
    Zero_mean_matrix[:, column] = zm
    zm_images = zm.resize(64, 64)
    zero_mean.append(zm)
    column = column + 1

# print('zero mean')
gridDisplay(zero_mean, 'zero mean')

d = (np.dot(np.transpose(Zero_mean_matrix), Zero_mean_matrix)) / 64
p_list = []
w2, v2 = la.eigh(d)

for ev in v2:
    ev_transpose = np.transpose(np.matrix(ev))
    p = np.dot(Zero_mean_matrix, ev_transpose)
    # norms = np.la.norm(u, axis=0)
    p = p / np.linalg.norm(p)
    #     minu = np.min(u)
    #     maxu = np.max(u)
    #     u = u-float(minu)
    #     u = u/float((maxu-minu))
    p_i = p.reshape(64, 64)
    p_list.append(p_i)

# print('eigen faces')
gridDisplay(p_list, 'eigen faces')

dict = {}


def Reconstruct(k, boolean, title):
    weights = np.zeros((170, k))
    matrixU = np.zeros((4096, k))
    c = 0
    for val in range(k - 1, -1, -1):
        matrixU[:, c] = p_list[val].flatten()
        c = c + 1
    rec_face = []
    for face_num in range(0, 170):
        w = np.dot(np.transpose(matrixU), Zero_mean_matrix[:, face_num])
        # w = Zero_mean_matrix[:,face_num]*np.transpose(matrixU)
        weights[face_num, :] = w
        # face=np.zeros((1, 4096))
        #         face = np.dot(w[0], matrixU[:,0])
        #         for i in range(1,k):
        #             face = face + np.dot(w[i], matrixU[:,i])
        #         #print(face.shape)
        #         face = face+np.transpose(mean)

        face = np.dot(w, np.transpose(matrixU))
        minf = np.min(face)
        maxf = np.max(face)
        face = face - float(minf)
        face = face / float((maxf - minf))
        # face = face + np.transpose(mean)
        reshape_face = face.reshape(64, 64)
        rec_face.append(reshape_face)
    arr = np.reshape(rec_face, (170, 64 * 64))
    arr = arr.transpose()
    mse = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            mse += [((A[i, j] - arr[i, j]) ** 2)]
    mse = np.array(mse)
    print(np.mean(mse))
    if boolean is True:
        gridDisplay(rec_face, title)
    dict[k] = weights


Reconstruct(1, True, 'for k = 1')

Reconstruct(30, True, 'for k = 30')

Reconstruct(120, True, 'for k = 120')
