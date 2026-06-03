# Citing CASTLE

If you use CASTLE in your research, please cite:

## BibTeX

```bibtex
@article{CASTLE,
  title={CASTLE: a training‑free foundation‑model pipeline for unsupervised, cross‑species behavioral classification},
  author={Liu, Yu-Shun and Yeh, Han-Yuan and Hu, Yu-Ting and Wu, Bing-Shiuan and Chen, Yi-Fang and Yang, Jia-Bin and Jasmin, Sureka and Hsu, Ching-Lung and Lin, Suewei and Chen, Chun-Hao and Wu, Yu-Wei},
  journal={bioRxiv},
  year={2025}
}
```

## APA Format

Liu, Y.-S., Yeh, H.-Y., Hu, Y.-T., Wu, B.-S., Chen, Y.-F., Yang, J.-B., Jasmin, S., Hsu, C.-L., Lin, S., Chen, C.-H., & Wu, Y.-W. (2025). CASTLE: A training-free foundation-model pipeline for unsupervised, cross-species behavioral classification. *bioRxiv*.

## Paper

📄 [Read on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.22.671685v2)

---

## Component Citations

CASTLE builds on these foundational works. Please also consider citing them:

### SAM (Segment Anything Model)

```bibtex
@inproceedings{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  booktitle={ICCV},
  year={2023}
}
```

### DeAOT (Decoupling Features for Video Object Segmentation)

```bibtex
@inproceedings{yang2022decoupling,
  title={Decoupling Features in Hierarchical Propagation for Video Object Segmentation},
  author={Yang, Zongxin and Yang, Yi},
  booktitle={NeurIPS},
  year={2022}
}
```

### DINOv2 / DINOv3

CASTLE's default visual encoder is now **DINOv3** (`dinov3_vitb16`); DINOv2 (`dinov2_vitb14_reg4_pretrain`) remains available as a selectable alternative. If you used the DINOv2 encoder, please cite:

```bibtex
@article{oquab2024dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal={TMLR},
  year={2024}
}
```

!!! note "Citing DINOv3"
    When you use the default DINOv3 encoder, please also cite the DINOv3 paper. See the [DINOv3 release](https://github.com/facebookresearch/dinov3) for the canonical BibTeX entry.

### UMAP

```bibtex
@article{mcinnes2018umap,
  title={UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author={McInnes, Leland and Healy, John and Melville, James},
  journal={arXiv preprint arXiv:1802.03426},
  year={2018}
}
```
