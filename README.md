# runway_for_ml


# Project Statement

We recognize that despite the emergence of deep Learning frameworks such as `pytorch` and higher-level frameworks such as `pytorch lightning` that separates the concerns of data, model, training, inference, and testing. There are still needs for yet a higher-level framework that addresses **data preparation**, **experiments configuration**, **systematic logging**, and **working with remote GPU clusters** (e.g. HPC). These common, indispensable functionalities are often re-implemented for individual projects, which hurt reproducibility, costs precious time, and hinders effective communications between ML developers.

We introduce **Runway**, a ML framework that delivers the last-mile solution so that researchers and engineers can focus on the essentials. In particular, we aim to 
1. Provide a **configurable data processing pipeline** that is easy to inspect and manipulate.
2. Provide an **experiment configuration system** so that it is convenient to conduct experiments in different settings without changing the code.
3. Provide a **systematic logging system** that makes it easy to log results and manage experiments both locally or on online platforms (e.g. weights-and-bias)
4. Provide a set of **tools that simplifies training/testing on remote GPU clusters** (e.g. HPC/Multiple GPU training)

With *Runway*, we hope to help ML researchers and engineers focus on the essential part of machine learning - data processing, modeling, inference, training, and evaluation. Our goal is to build a robust and flexible framework that gives developers complete freedom in these essential parts, while removing the tedious book-keeping. The philosophy, if taken to the extreme, entails that every line of code should reflect a design decision, otherwise there should be a configuration or a ready-to-use tool available. 

# Workflow

## Installation

Install with pip: `pip install runway_for_ml`

Alternatively, you can add runway as a submodule for more flexibility by running the following command

```bash
git submodule add git@github.com:EriChen0615/runway_for_ml.git runway_for_ml
git commit -m "Added runway_for_ml to the project"
```

## Initialize Runway Project

Change into target directory and run `runway init` to initialize the project. This would give you the following files/folders:

```
- data (where data is stored)
- local_cache (default caching location)
- third_party (where third party code goes)
- experiments (where experiments are stored)
- configs
    - meta_config.libsonnet
    - data_config.libsonnet
    - model_config.libsonnet
- data_processing
    - data_transforms.py
    - feature_loaders.py
- executors
    - custom_executor.py
- modeling
    - custom_modeling.py
- metrics 
    - custom_metric.py
```

## Coding Up Your Project

After setting up the runway project, you are ready to code! See [Development Manual](#-Development-Manual) for detail on how to develop code and test as you go with runway.

## Training 

Runway is organized in *experiments*. Each experiment corresponds to one trained model and possibly multiple inferences/evaluations. 

To begin training, you will need to:

1. Create an experiment configuration file using `runway-experiment <exp_name>`
2. Run training using `runway-train <exp_name>`


You can initialize a new experiment from existing ones, using `runway-experiment <exp_name> --from <existing_exp_name>`. The experiment configuration file will be cloned.

The two steps can be combined into one if you just want to change a few parameters of an existing experiment: `runway-train <exp_name> --from <existing_exp_name> --opts <param=value> ...`

## Inference & Evaluation

To run **inference**, you will need to specify an existing experiment with trained models. This can be done by `runway-infer <exp_name> <infer_suffix>`, where `<infer_suffix>` will be appended to `<exp_name>` to identify an inference run. 

You can also use `--opts` to override configurations, or use `--config <config_file>` to specify the configuration file to use for inference.

By default, evaluation will be run together with the inference. If you want to run evaluation separately, you can use `runway-eval <infer_run_name> --opts ...`

## Train/Infer/Evaluate in One Command

Runway provides a helper command that combines the above steps: `runway-run <exp_name> --from <existing_exp_name> --opts ...`. This will sequentially call `runway-train` and `runway-infer`. 

# Development Manual

Runway provides a framework to separate **data**, **modeling**, **training/inference**, and **evaluation**.

## Data

There are two aspects with data in ML stack - **loading** and **preprocessing**, which are handled by *feature loaders* and *data transforms*, respectively. You can configure and connect *feature loaders* and *data transformers* to form a *data pipeline* in the configuration file (usually `data_config.libsonnet`, or a custom jsonnet file).

### Declaring Feature Loaders

A *feature loader* load the dataset. It is a function decorated by `@register_to(FeatureLoader_Registry`. The function must take `feature_names` and `split` as parameters. The return type of the function is not restricted so long as the following data transforms can operate on it.

An example is given below:

```python
@register_to(FeatureLoader_Registry)
def LoadGEMSGDDataset(feature_names, split='train'):
    feature_dict = defaultdict(EasyDict) 
    ds = load_dataset('gem', 'schema_guided_dialog', split=split)
    feature_dict = keep_ds_columns(ds, feature_names)
    return feature_dict
```

> The necessary imports are already given at `runway-init` call

### Declaring Data Transforms

A data transform is a functor (i.e., a class whose object is callable) that takes in data, process it, and then return it. There are no restrictions on the input/output types.

Runway provides a few super-classes to help define data transforms that handles specific input type. For example, the `HFDatasetTransform` base class provides functionalities that work with HuggingFace's `datasets.Dataset` objects. Other useful base classes include `RowWiseTransform` that apply the same transformation to every row. If there is no suitable base class to inherit from, the functor should inherit `runway_for_ml.data_transforms.BaseTransform`.

You can override the following methods in the functor to implement your data transform:

- `_call(self, ...)`: the argument lists depends on your superclass. The data processing logic should reside here.
- `_preprocess(self, data)`: preprocess data for transform. Can be used to handle edge cases/unify interfaces etc.
- `setup(self, *args, **kwargs)`: where the functor is configured. E.g., changing transform parameters.
- `_check_input(self, data)`: optional input checking
- `_check_output(self, data)`: optional output checking
- `_apply_mapping(self, data, in_out_col_mapping)`: this enables data field selection. May be defined in superclass.

### Forming data pipeline

A *data pipeline* is defined in the configuration file. The key components are:

```json
{
    "name": "name of pipeline",
    "regenerate": true/false,
    "do_inspect": true/false,
    "inspector_config": {},
    "in_features": [ 
        {
        "feature_names": ["feature1", "feature2"],
        "feature_loader": "base_feature_loader",
        "splits": ["train", "test", "valid"], // the splits available
        "use_cache": true,
        },
        ...
    ],
    "transforms": {
        "train": train_transforms,
        "test": test_transforms,
        "valid": valid_transforms,
    },
}
```

## Modeling








