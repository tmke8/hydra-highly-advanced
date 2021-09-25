# Highly advanced example of using hydra

Here, we make use of `attr.s` to turn regular classes into config classes for hydra.
We then use `OmegaConf.to_object(...)` to instantiate these classes.
Except for the `to_object()` call, most interesting things happen in `model.py`.

To run this example with default settings:

```
python run.py
```

Overriding the choice of data:
```
python run.py data=cmnist/no_padding
```
