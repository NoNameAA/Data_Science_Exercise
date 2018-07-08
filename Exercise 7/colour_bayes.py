import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from skimage import color as sc
from sklearn import pipeline as sp
from sklearn import preprocessing as spr


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 114, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 187, 187),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=71, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)
    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))
    # print(pixels)

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)

def RGB_LAB(arr):
    arr = arr.reshape(1, -1, 3)
    arr = sc.rgb2lab(arr)
    arr = arr.reshape(-1, 3)
    return arr

def main(infile):
    data = pd.read_csv(infile)
    X = data[['R', 'G', 'B']] / 255 # array with shape (n, 3). Divide by 255 so components are all 0-1.
    y = data['Label'] # array with shape (n,) of colour words.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    model_rgb = GaussianNB()

    model_rgb.fit(X_train, y_train)
    accuracy_score = model_rgb.score(X_test, y_test)
    print("The accuracy score of RGB is %.3g" % accuracy_score)
    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')

    # TODO: build model_rgb to predict y from X.
    # TODO: print model_rgb's accuracy_score

    model_lab = sp.make_pipeline(
        spr.FunctionTransformer(RGB_LAB),
        GaussianNB(priors=None)
    )
    model_lab.fit(X_train, y_train)
    accuracy_score = model_lab.score(X_test, y_test)
    print("The accuracy score of LAB is %.3g" % accuracy_score)
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')

    # print(lab_X)

    # TODO: build model_lab to predict y from X by converting to LAB colour first.
    # TODO: print model_lab's accuracy_score


if __name__ == '__main__':
    main(sys.argv[1])
