import pandas as pd
import numpy as np
import matplotlib
from math import sqrt
import plotnine as p9

matplotlib.use("agg") #this will disable the plot preview, so use .save
file_path = 'd:/home/data.test.gz'
output_file_path = 'D:/home/hw2_op.png'
data_file = pd.read_csv(file_path,sep=' ',header = None)
data_file.shape

single_intensity_vector = data_file.iloc[1,1:]
n_pixels = int(sqrt(len(single_intensity_vector)))
single_image = pd.DataFrame({
    'row':np.flip(np.repeat(np.arange(n_pixels),n_pixels)),
    'col':np.tile(np.arange(n_pixels),n_pixels),
    'intensity':single_intensity_vector
})
single_image_plot = p9.ggplot()+\
    p9.geom_tile(
        p9.aes(
            x='col',
            y='row',
            fill='intensity'
        ),single_image
    )+\
    p9.scale_fill_gradient(
        low='black',
        high='white'
    )
#single_image_plot


list_of_images = []
for image_number in range(12):
    intensity_vector  =data_file.iloc[image_number,1:]
    n_pixels = int(sqrt(len(intensity_vector)))
    image_label = data_file.iloc[image_number,0]

    one_image_df =pd.DataFrame({
        'observation':image_number,
        'label':image_label,
        'row':np.flip(np.repeat(np.arange(n_pixels),n_pixels)),
        'col':np.tile(np.arange(n_pixels),n_pixels),
        'intensity':intensity_vector
    })
    list_of_images.append(one_image_df)

several_images_df = pd.concat(list_of_images)

myplot = p9.ggplot()+\
    p9.facet_wrap(["observation","label"],labeller="label_both")+\
    p9.geom_tile(#for drawing squares
        p9.aes(
            x='col',
            y='row',
            fill='intensity'
        ),
        data = several_images_df
    )+\
    p9.scale_fill_gradient(
        low='black',
        high='white'
    )
print(myplot)
p9.ggsave(myplot,output_file_path)
