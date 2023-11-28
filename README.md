# FairLISA
Code for the NeurIPS'2023 paper "FairLISA: Fair User Modeling with Limited Sensitive Attributes Information"

# Run

python3 run.py -DATA your_dataset -FILTER_MODE separate -FAIRNESS_RATIO 1.0 -FAIRNESS_RATIO_NOFEATURE 0.5 -CUDA 5 -USER_NUM 358415 -ITEM_NUM 183 -KNOWLEDGE_NUM 16 -LATENT_NUM 16 -MODEL IRT -NO_FEATURE 0.6 -USE_NOFEATURE True


## BibTex
Please cite this paper if you use our codes. Thanks!

```
@inproceedings{zhang2023fairlisa,
  title={FairLISA: Fair User Modeling with Limited Sensitive Attributes Information},
  author={Zhang, Zheng and Liu, Qi and Jiang, Hao and Wang, Fei and Zhuang, Yan and Wu, Le and Gao, Weibo and Chen, Enhong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
