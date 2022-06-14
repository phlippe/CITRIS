## Data generation

This folder contains files with which the datasets of this paper can be generated.

### (Instantaneous) Temporal Causal3DIdent

The dataset generation for the (Instantaneous) Temporal Causal3DIdent dataset contains multiple steps, and hence needs multiple files. We summarize them in the folder [`temporal_causal3dident`](temporal_causal3dident/). The Blender processing was adapted from https://github.com/brendel-group/cl-ica.

The first step is to generate a sequence of latent factors, which is implemented in [`data_generation_causal3dident.py`](temporal_causal3dident/data_generation_causal3dident.py). These latents are then taken as input to [`generate_causal3d_images.py`](temporal_causal3dident/generate_causal3d_images.py), which iterates over the time steps and renders according images using Blender. These images are then stacked and saved in a compressed numpy file.

#### Requirements

* Install Blender 2.93.8 (newer versions may not be supported)
* Download the [`data`](https://github.com/brendel-group/cl-ica/tree/master/tools/3dident/data) folder and place in `temporal_causal3dident/data/`
* Add the blender files for the 6 additional shapes ([Armadillo](http://graphics.stanford.edu/data/3Dscanrep/), [Bunny](http://graphics.stanford.edu/data/3Dscanrep/), [Cow](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/#spot), [Dragon](http://graphics.stanford.edu/data/3Dscanrep/), [Head](https://gfx.cs.princeton.edu/proj/sugcon/models/), [Horse](https://www.cc.gatech.edu/projects/large_models/horse.html)) to the folder `temporal_causal3dident/data/shapes/`. The shapes may have to be converted to Blender files first, and rescaled to a suitable size first. Please respect the licenses of the individual shapes. We do not include these shapes here due to the licenses. However, if interested, we can provide them upon request (please send a mail to p.lippe@uva.nl).

### Interventional Pong

The dataset generation for the Interventional Pong dataset is implemented in the file [`data_generation_interventional_pong.py`](data_generation_interventional_pong.py). It is mainly splitted into two parts: 
1. The latent dynamical causal system which implements the game engine of Pong. It first creates a numpy file that contains a long sequence of causal factor values which corresponds to this system.
2. The image rendering. For this, we use matplotlib and map each time step of causal factors to an image. 

For efficiency, the second step is implemented with multiprocessing, which brings down the generation time to 10-15 minutes, depending on the used CPU.

### Ball-in-Boxes

The dataset generation for the Ball-in-Boxes dataset (appendix of the CITRIS paper) is implemented in the file [`data_generation_ball_in_boxes.py`](data_generation_ball_in_boxes.py). It is structured in the same way as the Interventional Pong dataset, and uses matplotlib to create the images. 


### Voronoi benchmark

The dataset generation for the Voronoi benchmark is implemented in the file [`data_generation_voronoi.py`](data_generation_voronoi.py). We use the same file to generate various graph structures and the random neural network distributions. Similar to the previous datasets, matplotlib is used to render the images.

### Causal Pinball

The dataset generation for the Causal Pinball dataset is implemented in the file [`data_generation_pinball.py`](data_generation_pinball.py), including the game dynamics with physical collisions and more. While there can occur rare scenarios where the collision calculation is not perfect (e.g. colliding with two objects of opposite direction at the same time), we have not experienced this to be a problem in both training and normal game play dynamics. Note that compared to the other matplotlib datasets, the Causal Pinball dataset is slower to generate since we use a higher resolution (64x64) and several objects need to be rendered.