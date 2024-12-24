# SlimIPL Implementation  
Implementation of the [SlimIPL](https://arxiv.org/abs/2010.11524) algorithm in Pytorch Lightning. If you want to use other optimizer and scheduler change configuration in [config.yaml](https://github.com/NikitaKlichko/slimipl_implementation/blob/main/config.yaml), **configure_optimizers** in [slimipl.py](https://github.com/NikitaKlichko/slimipl_implementation/blob/main/src/slimipl.py) and **create_optimizer**  and **create_scheduler** in [train.py](https://github.com/NikitaKlichko/slimipl_implementation/blob/main/train.py#L34).

Launch  example:
```
python train.py --config config.yaml
```