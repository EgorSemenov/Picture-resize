import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def img_show(img, title='', str=''):
    plt.figure(dpi=200)
    plt.imshow(img, cmap='plasma')
    title = title + str
    plt.title(title)
    plt.show()


def bilinear_resize_vectorized(image, height, width):
    """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
    img_height, img_width = image.shape[0], image.shape[1]

    image = image.ravel()

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    y, x = np.divmod(np.arange(height * width), width)

    x_l = np.floor(x_ratio * x).astype('int32')
    y_l = np.floor(y_ratio * y).astype('int32')

    x_h = np.ceil(x_ratio * x).astype('int32')
    y_h = np.ceil(y_ratio * y).astype('int32')

    x_weight = (x_ratio * x) - x_l
    y_weight = (y_ratio * y) - y_l

    a = image[y_l * img_width*3 + x_l*3]
    a1 = image[y_l * img_width * 3 + x_l*3 + 1]
    a2 = image[y_l * img_width*3 + x_l*3 + 2]
    b = image[y_l * img_width*3 + x_h*3]
    b1 = image[y_l * img_width*3 + x_h*3 + 1]
    b2 = image[y_l * img_width*3 + x_h*3 + 2]
    c = image[y_h * img_width*3 + x_l*3]
    c1 = image[y_h * img_width*3 + x_l*3 + 1]
    c2 = image[y_h * img_width*3 + x_l*3 + 2]
    d = image[y_h * img_width*3 + x_h*3]
    d1 = image[y_h * img_width*3 + x_h*3 + 1]
    d2 = image[y_h * img_width*3 + x_h*3 + 2]

    resized = a * (1 - x_weight) * (1 - y_weight) + \
              b * x_weight * (1 - y_weight) + \
              c * y_weight * (1 - x_weight) + \
              d * x_weight * y_weight
    resized = resized.astype(int)
    resized1 = a1 * (1 - x_weight) * (1 - y_weight) + \
               b1 * x_weight * (1 - y_weight) + \
               c1 * y_weight * (1 - x_weight) + \
               d1 * x_weight * y_weight
    resized1 = resized1.astype(int)
    resized2 = a2 * (1 - x_weight) * (1 - y_weight) + \
               b2 * x_weight * (1 - y_weight) + \
               c2 * y_weight * (1 - x_weight) + \
               d2 * x_weight * y_weight
    resized2 = resized2.astype(int)
    resized = np.stack((resized, resized1, resized2), axis=1)

    return resized.reshape(height, width, 3)


img = mpimg.imread('clear_img.jpg')
img_show(img, 'исходная картинка', '')
new_img = bilinear_resize_vectorized(img, int(img.shape[0]*3.5//1), int(img.shape[1]*3.5//1))
img_show(new_img, 'увеличенная картинка')
new_img = bilinear_resize_vectorized(new_img, int(new_img.shape[0]/3.5//1), int(new_img.shape[1]/3.5//1))
img_show(new_img, 'уменьшенная назад картинка')
result = np.absolute(new_img - img)
img_show(result, 'разница по модулю(чем темнее цвет, тем она меньше)')
print(result)