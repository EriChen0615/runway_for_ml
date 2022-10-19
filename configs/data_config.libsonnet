// This is the configuration file for dataloaders. It registers what dataloaders are available to use
// For each dataloader, it also registers what dataset modules are available to obtain processed features
// All dataloader and feature loaders must be declared here for runway to work

// data path configuration
// local default_cache_folder = '../data/ok-vqa/cache'; // override as appropriate

// Configurations for feature loaders, define as appropriate
// local example_feature_config = { // usable in ExampleLoadFeatures function
//   train: "FILE LOCATION OF TRAINING DATA",
//   test: "FILE LOCATION OF TESTING DATA",
// };

local example_feature_loader = {
  name: "YOUR FEATURE_LOADER CLASS NAME",
  kwargs: { // arguments to feature_loader init
    train: "FILE LOCATION OF TRAINING DATA",
    test: "FILE LOCATION OF TEST DATA"
  },
  cache_data: true,
  use_cache: true,
};

// local example_data_source = {
//   data_source_class: "YOUR DATA_SOURCE CLASS NAME",
//   data_source_args: { // arguments to data_source init
//   },
//   features: [ // define the `columns` of data_source
//     {
//       feature_name: "loader_name", 
//       feature_loader: example_feature_loader,
//       splits: ["train", "test", "valid"],
//     },
//   ],
// };

local default_dataloader_args = {
  batch_size: 1,
  shuffle: false,
  sampler: null,
}; // see https://pytorch.org/docs/stable/data.html for arguments

local example_data_pipeline = {
  name: 'example_data_pipeline',
  regenerate: true,
  dataloader_args: {
    train: default_dataloader_args {
      shuffle: true // override
    },
    test: default_dataloader_args {

    },
    valid: default_dataloader_args {

    },
  },
  in_features: [ // features used by the pipelines (MUST BE available at init)
    {
      feature_name: "in_feature_name",
      feature_loader: example_feature_loader,
      splits: ["train", "test", "valid"], // the splits available
      use_cache: true,
    },
  ],

  // define transform for each split
  local train_transforms = [
    {
      type: "Transform function name 1",
      use_features: [],
      kwargs: {},
      out_features: ["col1", "col2"], // override col1; if 'col1+', result will be appended to col1
      batched: 0,
    },
    {
      type: "Transform function name 2",
      use_features: ["col2"], // out_feature in previous transform is available
      kwargs: {},
      out_features: ["col2+"], // override col1; if 'col1+', result will be appended to col1
      batched: 1,
    },
  ],
  local test_transforms = [
    {
      name: "Transform function name",
      use_features: [],
      kwargs: {},
      out_features: ["col2"],
      batched: 1,
    },
  ],
  local valid_transforms = train_transforms,
  transforms: {
    train: train_transforms,
    test: test_transforms,
    valid: valid_transforms,
  },
};

{
  "example_data_pipeline": example_data_pipeline
}