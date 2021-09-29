# Highly advanced example of using hydra

Here, we make use of `attr.s` to turn regular classes into config classes for hydra.
We then use `OmegaConf.to_object(...)` to instantiate these classes.
Except for the `to_object()` call, most interesting things happen in `model.py`.

### Note
When combining `to_object()` and `instantiate()`, a few rules have to be followed
(but there is an open issue to make these rules unnecessary).
First, the top level configuration object may not be marked for instantiation
(that is, it may not have a `_target_` attribute).
Second, classes that are to be instantiated may not take structured configs (i.e., dataclasses or attrs classes)
as arguments.

## Usage

To run this example with default settings:

```
python run.py
```

Overriding the choice of data:
```
python run.py data=cmnist/no_padding
```
