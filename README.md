## About

Repository for the development of *IntelliPore: A Foundation Model for Gas Adsorption in Porous Materials*.

<p align="center">
  <img alt="RetNeXt feature maps" src="https://raw.githubusercontent.com/adosar/intellipore-paper/master/images/toc.png" width="65%"/>
</p>

By leveraging 12.5 million publicly available adsorption data from MOFs and COFs, IntelliPore provides:

* **Transferable representations** across adsorption tasks and porous material classes
* **Few-shot adaptation**, substantially reducing the amount of of required downstream labeled data
* **State-of-the-art performance** for gas adsorption property prediction
* **Exceptional parameter efficiency** with only 0.86 million parameters, enabling efficient fine-tuning and inference

## Resources

* The pretrained model is distributed through the [AIdsorb package](https://aidsorb.readthedocs.io/en/latest/_autosummary/aidsorb.modules.voxels.html#aidsorb.modules.voxels.IntelliPore)
* A complete example to fine-tune the model is available at the [AIdsorb Gallery](https://aidsorb.readthedocs.io/en/latest/auto_examples/finetune.html)
* All the fine-tuned models from the benchmarks are available at: https://github.com/adosar/intellipore-models
* Online demo is available at: https://aidsorb-online.streamlit.app/Predict_with_energy_images

## Performance summary
Comparison of IntelliPore and other baseline models for gas adsorption prediction in terms of $R^2$ (higher is better). 

**Best** model is highlighted with **bold**.

| Adsorption Property                                                                 | CGCNN   | MOFormer   | PMTransformer   | SpbNet    | IntelliPore |
|:------------------------------------------------------------------------------------|:--------|:-----------|:----------------|:----------|:--------------|
| CO₂ uptake at 298 K and 2.5 bar (MOFs)                                              | 0.605   | 0.563      | 0.886           | **0.944** | 0.943         |
| CH₄ uptake at 298 K and 2.5 bar (MOFs)                                              | 0.667   | 0.608      | 0.853           | 0.924     | **0.960**     |
| N₂ uptake at 298 K and 0.9 bar (MOFs)                                               | 0.634   | 0.562      | 0.777           | 0.874     | **0.906**     |
| Xe binary uptake at 273 K and 10 bar (MOFs)                                         | 0.697   | 0.675      | 0.901           | 0.926     | **0.946**     |
| Kr binary uptake at 273 K and 10 bar (MOFs)                                         | 0.733   | 0.736      | 0.842           | 0.856     | **0.899**     |
| H₂ uptake at 77 K and 2 bar (MOFs)                                                  | 0.621   | 0.627      | 0.932           | 0.975     | **0.985**     |
| CO₂ heat of adsorption at 298 K and 2.5 bar (MOFs)                                  | 0.812   | 0.755      | 0.906           | 0.903     | **0.909**     |
| H₂ heat of adsorption at 77 K and 2 bar (MOFs)                                      | 0.798   | 0.711      | 0.895           | 0.905     | **0.926**     |
| CH₄ uptake at 298 K and 5.8 bar (COFs)                                              | 0.516   | 0.566      | 0.909           | 0.952     | **0.989**     |
| CH₄ uptake at 298 K and 65 bar (COFs)                                               | 0.556   | 0.541      | 0.967           | 0.976     | **0.987**     |
| CH₄ uptake at 298 K and 1 bar (PPNs)                                                | 0.293   | 0.417      | 0.559           | 0.890     | **0.973**     |
| CH₄ uptake at 298 K and 65 bar (PPNs)                                               | 0.692   | 0.636      | 0.942           | 0.968     | **0.987**     |
| CH₄ unitless Henry coefficient at 298 K (ZEOs)                                      | 0.237   | 0.107      | 0.435           | 0.809     | **0.910**     |
| CH₄ isosteric heat of adsorption at 298 K (ZEOs)                                    | 0.411   | 0.388      | 0.836           | 0.935     | **0.973**     |
| C₃H₈/C₃H₆ selectivity at 298 K and 1 bar (MOFs)                                     | 0.020   | -0.010     | 0.401           | 0.317     | **0.742**     |
| C₃H₈/C₃H₆ infinite dillution selectivity at 298 K (MOFs)                            | 0.293   | 0.201      | 0.599           | 0.627     | **0.916**     |
| C₃H₆ binary uptake at 298 K and 1 bar (MOFs)                                        | 0.727   | 0.602      | 0.824           | 0.900     | **0.942**     |
| C₃H₈ binary uptake at 298 K and 1 bar (MOFs)                                        | 0.736   | 0.689      | 0.875           | 0.923     | **0.946**     |
| $N_\mathrm{C_3H_8} \times \log(S_\mathrm{C_3H_8/C_3H_6})$ at 298 K and 1 bar (MOFs) | 0.358   | 0.266      | 0.528           | 0.630     | **0.816**     |
| C₃H₈ log Henry coefficient at 298 K (MOFs)                                          | 0.239   | 0.064      | 0.337           | 0.351     | **0.873**     |
| C₃H₆ log Henry coefficient at 298 K (MOFs)                                          | 0.299   | 0.192      | 0.481           | 0.647     | **0.898**     |
| CH₄/N₂ selectivity and 298 K and 0.1 bar (MOFs)                                     | 0.752   | 0.286      | 0.823           | 0.866     | **0.885**     |
| CH₄/N₂ selectivity and 298 K and 10 bar (MOFs)                                      | 0.728   | 0.723      | 0.912           | 0.947     | **0.962**     |
| CH₄/N₂ selectivity and 298 K and 1 bar (MOFs)                                       | 0.718   | 0.698      | 0.905           | 0.949     | **0.968**     |
| CH₄ binary uptake at 298 K and 0.1 bar (MOFs)                                       | 0.523   | 0.452      | 0.714           | 0.900     | **0.951**     |
| CH₄ binary uptake at 298 K and 10 bar (MOFs)                                        | 0.701   | 0.707      | 0.815           | 0.929     | **0.935**     |
| CH₄ binary uptake at 298 K and 1 bar (MOFs)                                         | 0.496   | 0.617      | 0.769           | 0.918     | **0.951**     |
| N₂ binary uptake at 298 K and 0.1 bar (MOFs)                                        | 0.550   | 0.588      | 0.710           | 0.869     | **0.911**     |
| N₂ binary uptake at 298 K and 10 bar (MOFs)                                         | 0.792   | 0.814      | 0.886           | **0.943** | **0.943**     |
| N₂ binary uptake at 298 K and 1 bar (MOFs)                                          | 0.617   | 0.661      | 0.783           | **0.924** | 0.905         |



## Citing

If you use any of these models in your research, please cite:

```bibtex
@article{intellipore2025,
  title   = {IntelliPore: A Foundation Model for Gas Adsorption in Porous Materials},
  author  = {Author, A. and Author, B. and ...},
  journal = {Journal},
  year    = {2025},
  doi     = {TBD}
}
```
