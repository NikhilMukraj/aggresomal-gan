import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


path = os.getcwd()
plt.imshow(mpimg.imread(path + '\\test results v2.png'))
plt.show()