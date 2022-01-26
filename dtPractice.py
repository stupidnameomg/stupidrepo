# SETUP
import deeptrack as dt
import numpy as np
import matplotlib.pyplot as plt
from deeptrack.extras import datasets

import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers


datasets.load('QuantumDots')

image_size = 128

#DEFINING DATASETS
# training set: simulated 128x128 px images w multiple particles each
# dots are sampled w normal distribution w sd 5 px

#create particles
particle = dt.PointParticle(
    position=lambda: np.random.rand(2) * image_size,
    z=lambda: np.random.standard_normal() * 5,  # trying this since this is better according to doc on randn
    intensity=lambda: 1 + np.random.rand() * 9,
    position_unit="pixel",
)

no_particles = lambda: np.random.randint(10, 20)
particles = particle ** no_particles #

# define imageing
optics = dt.Fluorescence(
    NA=lambda: 0.6 + np.random.rand() * 0.2,
    wavelength=500e-9,
    resolution=1e-6, # Fråga till Daniel?
    magnification=10,
    output_region=(0, 0, image_size, image_size) # why 0, 0?
)
#normalize, not sure if this should be done final or here?
normalization = dt.NormalizeMinMax(
    min=lambda: np.random.rand() * 0.4,
    max=lambda min: min + 0.1 + np.random.rand() * 0.5
)
# add noise within ratio
noise = dt.Poisson(
    snr=lambda: 4 + np.random.rand() * 3,
    background=lambda: normalization.min
)

# define combination
imaged_particle = optics(particles)
# why do we flip the images?
dataset = dt.FlipUD(dt.FlipLR(imaged_particle + normalization + noise))

# defining the training label: each particle repr. disk w r=3px
circle_radius = 3

X, Y = np.mgrid[:2*circle_radius, :2*circle_radius]
circle = (X-circle_radius+0.5)**2 + (Y-circle_radius+0.5)**2 < circle_radius**2
circle = np.expand_dims(circle, axis=-1)

get_mask = dt.SampleToMasks(
    lambda: lambda image: circle, # samma här?
    output_region=lambda:optics.output_region, #  vad händer här?
    merge_method="or"
)


def get_label(image_of_particles):
    return get_mask.update().resolve(image_of_particles)


# VISUALIZING THE DATASET
# plot 16 images w green circle indication particle position
no_images = 16

for _ in range(no_images):
    plt.figure(figsize=(15, 5))
    dataset.update()
    image_of_particle = dataset.resolve()

    position = np.array(image_of_particle.get_property("position", get_one=False))
    plt .imshow(image_of_particle[..., 0], cmap="gray")
    plt.scatter(position[:, 1], position[:, 0])
    plt.show()

# DEFINING NETWORK
loss = dt.losses.flatten(
    dt.losses.weighted_crossentropy()
)