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
  feature_loader_class: "YOUR FEATURE_LOADER CLASS NAME",
  feature_loader_args: { // arguments to feature_loader init
    train: "FILE LOCATION OF TRAINING DATA",
    test: "FILE LOCATION OF TEST DATA"
  },
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

local example_data_pipeline = {
  data_pipeline_class: "YOUR DATA_PIPELINE CLASS NAME",
  data_pipeline_args: { // arguments to data_pipeline_args
  },
  in_features: [ // features used by the pipelines (MUST BE available at init)
    {
      feature_name: "in_feature_name",
      feature_loader: example_feature_loader,
      splits: ["train", "test", "valid"], // the splits available
      use_cache: 1,
    },
  ],

  // define transform for each split
  local train_transforms = [
    {
      type: "Transform function name 1",
      use_features: [],
      args: {},
      out_features: ["col1", "col2"], // override col1; if 'col1+', result will be appended to col1
      batched: 0,
    },
    {
      type: "Transform function name 2",
      use_features: ["col2"], // out_feature in previous transform is available
      args: {},
      out_features: ["col2+"], // override col1; if 'col1+', result will be appended to col1
      batched: 1,
    },
  ],
  local test_transforms = [
    {
      type: "Transform function name",
      use_features: [],
      args: {},
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