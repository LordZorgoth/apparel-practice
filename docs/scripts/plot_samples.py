import os
from matplotlib import gridspec

os.chdir("../../code")
from load_data import *
X, y = load_train_data(58)
os.chdir("../docs/scripts")

first_three_indices_of_each_class = np.array(
    [np.argwhere(y[:, k] == True)[:3, 0] for k in range(10)])
class_names = {0: "Short-Sleeve Tops",
               1: "Pants",
               2: "Sweaters",
               3: "Dresses",
               4: "Coats",
               5: "Open-Toe Shoes",
               6: "Long-Sleeve Tops",
               7: "Closed-Toe Shoes",
               8: "Bags",
               9: "Boots"}
gs = gridspec.GridSpec(5, 7, width_ratios=[1]*3 + [0.5] + [1]*3)
fig = plt.figure(figsize=(6, 8))
fig.suptitle("Sample Images from Dataset",
             fontfamily="Merriweather", fontweight="bold", fontsize=18)
for k in range(10):
    for i in range(3):
        ax = fig.add_subplot(gs[k//2, k%2*4 + i])
        if i == 1:
            ax.set_title("Class %s: %s" % (k, class_names[k]),
                         fontfamily="Merriweather", fontweight=400, pad=9)
        plt.imshow(X[first_three_indices_of_each_class[k]][i, 0], cmap='gray')
        ax.axis('off')
        ax.text(0.5, -0.2,
                "train/%s.png" % first_three_indices_of_each_class[k, i],
                transform=ax.transAxes, horizontalalignment="center",
                fontfamily="Merriweather", fontweight=300, fontsize=8)
for subplot_row in range(5):
    ax = plt.subplot(gs[subplot_row, 3])
    ax.axis('off')
plt.savefig('../images/sample_images.png')
