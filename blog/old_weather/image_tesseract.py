from PIL import Image
import numpy as np

template = np.asarray(Image.open("/home/ggdhines/t.png"))
a,b = template.shape
# template = template.reshape((1,a*b))[0]
print template.argmax()
print template.max()
# template = np.logical_not(template)
# print template[801386]
print template[0]

image2 = Image.open("/home/ggdhines/Databases/old_weather/aligned_images/Bear-AG-29-1939-0245.JPG")
grey_image =  np.asarray(image2.convert('L'))

merged = grey_image & template
merged = merged*255
low_values_indices = merged < 80
merged[low_values_indices] = 0
print merged.max()
img = Image.fromarray(merged, 'L')
img.save('/home/ggdhines/my.png')
