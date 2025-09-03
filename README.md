# Flow Matching and Diffusion Models from scratch

Wanted to play with them to learn how they work, so here it is.
The python files have all the ingredients (loss, sampling method, architecture, training loop), and the notebook (which is too heavy right now...) examines several toy examples:
- A 2D Gaussian distribution
- A 2D Gaussian Mixture Model
- Conditioning the model on the class of the sample (which Gaussian the sample was sampled from)
- Classifier-free guidance
- MNIST with conditioning
- Cifar-10 (didn't run yet)

The notebook also includes some visuals that show the generation process.
So far, only ran flow matching.
