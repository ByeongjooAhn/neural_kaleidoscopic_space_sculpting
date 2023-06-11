# Neural Kaleidoscopic Space Sculpting

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://imaging.cs.cmu.edu/neural_kaleidoscopic_space_sculpting/)
[![Paper](https://img.shields.io/badge/Paper-CVPR2023-blueviolet)](https://imaging.cs.cmu.edu/neural_kaleidoscopic_space_sculpting/assets/neural_kaleidoscopic_cvpr2023.pdf)
[![Video](https://img.shields.io/badge/Video-YouTube-red)](https://www.youtube.com/watch?v=ccheBtcE3Ec)

<p align="center">
    <img width="100%" src="media/teaser.gif"/>
</p>

Welcome to the official codebase for the Neural Kaleidoscopic Space Sculpting paper, a project that presents a novel method for single-shot full-surround 3D reconstruction using a kaleidoscopic image.

["Neural Kaleidoscopic Space Sculpting"](https://imaging.cs.cmu.edu/neural_kaleidoscopic_space_sculpting/)\
[Byeongjoo Ahn](https://byeongjooahn.github.io/), Michael De Zeeuw, [Ioannis Gkioulekas](https://www.cs.cmu.edu/~igkioule/), and [Aswin C. Sankaranarayanan](https://users.ece.cmu.edu/~saswin/)\
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023

This repository is based on the [IDR](https://github.com/lioryariv/idr) repository, which serves as the foundation for our work.

## Usage Instructions
### Installation
Create and activate the conda environment using the provided environment.yml file. Run the following commands in your terminal:
```bash
conda env create -f environment.yml
conda activate nkss
```

### Dataset
You can download the required kaleidoscopic image and calibration data [here](https://www.dropbox.com/sh/g6q0h43y6xwxnx2/AACwqHQ8Z-QQE5MIi_5UzEz-a?dl=0). Please ensure to place it in the `data/` directory.

### Training
To initiate the training process, navigate to the code directory and run the following commands:

```bash
python training/exp_runner.py --conf ./confs/toy.conf
```

### Extracting meshed surface
Post-training, you can extract the meshed surface by executing the commands below:

```bash
python evaluation/eval.py --conf ./confs/toy.conf --resolution 200 --eval_levelset --scale 0.25 --eval_rendering
```

## Related Research

Our group has also explored other areas in imaging using mirrors. You might find these related works interesting:

* [Kaleidoscopic Structured Light (TOG 2021)](https://imaging.cs.cmu.edu/kaleidoscopic_structured_light/)
* [Wide-Baseline Light Fields using Ellipsoidal Mirrors (PAMI 2022)](http://imagesci.ece.cmu.edu/files/paper/2022/WideBaselineLF_PAMI22.pdf)

## Citation

If you find our work useful or inspiring, please consider citing our paper using the following Bibtex entry:

```bibtex
@inproceedings{ahn2023neural,
    author    = {Ahn, Byeongjoo and De Zeeuw, Michael and Gkioulekas, Ioannis and Sankaranarayanan, Aswin C.},
    title     = {Neural Kaleidoscopic Space Sculpting},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```
