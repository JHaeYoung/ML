# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# data


np.random.seed(1)
x = np.arange(10)
y= np.random.rand(10)
rng = np.random.default_rng()
y= rng.normal(size=10)

#그래프 출력
plt.figure(figsize=(3,2))
plt.plot(x,y)
plt.show()
# %%

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = np.asarray(Image.open('stinkbug.png'))
print(repr(img))

imgplot = plt.imshow(img)

# %%

lum_img = img[:, :, 0]
plt.imshow(lum_img)
# %%
plt.imshow(lum_img, cmap="hot")
# %%
imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')
# %%
plt.hist(lum_img.ravel(), bins=range(256), fc='k', ec='k')
# %%
