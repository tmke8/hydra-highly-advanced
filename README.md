# Highly advanced example of using hydra

Here, we use tricks like a custom `__new__` function to turn regular classes into config classes for hydra.
We then use `instantiate()` with `_convert_="object"` to instantiate these classes.
Except for the `instantiate()` call, most interesting things happen in `model.py`.

#### Note
When using `_convert_="object"`, a few rules have to be followed
(but there is an open issue to make these rules unnecessary).
First, the top level configuration object may not be marked for instantiation
(that is, it may not have a `_target_` attribute).
Second, classes that are to be instantiated may not take structured configs (i.e., dataclasses or attrs classes)
as arguments.

## Usage
Requires Python 3.10+ and `hydra-core` 1.3+.

The simplest valid invocation is this:

```
python run.py seed=0 data=cmnist
```

This uses the config values stored in `conf/data/celeba/gender.yaml`:
```
python run.py seed=1 data=celeba/gender
```

And this one uses `conf/trainer/4gpus.yaml`:
```
python run.py trainer=4gpus data=cmnist seed=0
```
