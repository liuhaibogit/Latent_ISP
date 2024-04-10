# Latent_ISP

This is an implementation of 'Solving Inverse Obstacle Scattering Problem with Latent Surface Representations" by Junqing Chen, Bangti Jin and Haibo Liu. See the paper [here](https://arxiv.org/abs/2311.07187).

# Citing Latent_ISP
If you use Latent_ISP in your research, please cite the [paper](https://arxiv.org/abs/2311.07187):
```
@misc{chen2023solving,
	title={Solving Inverse Obstacle Scattering Problem with Latent Surface Representations}, 
	author={Junqing Chen and Bangti Jin and Haibo Liu},
	year={2023},
	eprint={2311.07187},
	archivePrefix={arXiv},
	primaryClass={math.NA}
}
```

# How to Use Latent_ISP
The preprocessing code has the following requirements:
- [bempp-cl](https://bempp.com/)
- [DeepSDF](https://github.com/facebookresearch/DeepSDF)

To use a trained model to reconstruct explicit mesh representations of shapes from far-field data, run:
> python far_field_plane.py

# Team
Junqing Chen, Bangti Jin, Haibo Liu

# Acknowledgements

# License
