# [ICME 2025] Variance-Reduction Guidance: Sampling Trajectory Optimization for Diffusion Models
*[Shifeng Xu](https://www.linkedin.com/in/shifeng-xu-a1b93517/), 
 [Yanzhu Liu](https://openreview.net/profile?id=~Yanzhu_Liu2), 
 [Adams Wai-Kin Kong](https://personal.ntu.edu.sg/AdamsKong/)*

Official Github Repo for Variance-Reduction Guidance: Sampling Trajectory Optimization for Diffusion Models

<img src="docs/fig_bedroom-compare-with-inter-steps.png" alt="">
Sampling process of DPM-Solver with order=2, schedule= *quadratic* and steps=10. 
(a) Original and optimized trajectories. 
The former has noise level sequence as 
{0.995, 0.97, 0.88, 0.72, 0.48, 0.23, 0.075, 0.0137, 0.00118, 0.000040}, and the latter has 
{0.991, 0.95, 0.86, 0.69, 0.45, 0.22, 0.068, 0.0129, 0.00121, 0.000054}. 
(b) Column 1 is generated images, column 6 is the initial Gaussian noises, 
and columns 2 - 5 are the intermediate results of the sampling progress. 

## Abstract
Diffusion models have become emerging generative models. 
Their sampling process involves multiple steps, 
and in each step the models predict the noise from a noisy sample. 
When the models make prediction, the output deviates from the ground truth, 
and we call such a deviation as *prediction error*. 
The prediction error accumulates over the sampling process and deteriorates generation quality. 
This paper introduces a novel technique for statistically measuring the prediction error 
and proposes the Variance-Reduction Guidance (**VRG**) method to mitigate this error. 
VRG does not require model fine-tuning or modification. 
Given a predefined sampling trajectory, it searches for a new trajectory 
which has the same number of sampling steps but produces higher quality results.
VRG is applicable to both conditional and unconditional generation. 
Experiments on various datasets and baselines demonstrate that 
VRG can significantly improve the generation quality of diffusion models. 

## Fundamental: The prediction error follows a Gaussian distribution.
Given a noisy sample and a timestep, diffusion models predict the noise within the sample. 
However, the prediction is not perfectly accurate and contains some error. 
Even in well-trained models, this error persists. Such prediction error follows Gaussian distribution.
<img src="docs/fig_pred_error_follow_gaussian.png" alt="">

## Our method
The sampling process of diffusion models consists of a sequence of iterative sampling steps. 
Between two adjacent steps, diffusion models not only transform samples but also transfer error. 
While the former is intended, the latter hinders the generation quality. 
Prediction error exists in every sampling step, and accumulates across sampling process. 

This paper introduces a novel technique for statistically measuring the prediction error and
proposes the Variance-Reduction Guidance (``VRG``) method to mitigate this error. 
Specifically, VRG optimizes sampling trajectory by adjusting the noise level of each sampling step. 
To achieve this, we employ a neural network to refine the trajectory 
by minimizing the variance of the prediction error. 
<img src="./docs/fig_VRG_network_flowchart.png" alt="" >

## How to run
The code of this project is self-contained and can be run directly without any additional files.
```code
python main.py         \
  --n_epochs   3000    \
  --lr         0.0002  \
  --lp         0.01    \
  --reg_lambda 100
```

## Citation
If this work is helpful for your research, please consider citing:

```
@inproceedings{icme2025_vrg_shifeng_xu,
    title={Variance-Reduction Guidance: Sampling Trajectory Optimization for Diffusion Models}, 
    author={Shifeng Xu, Yanzhu Liu, Adams Wai-Kin Kong},
    booktitle={ICME},
    year={2025},
}
```