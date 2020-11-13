`Version 0.0.1 <https://github.com/kijanac/luz/compare/2df600d...v0.0.1>`__
---------------------------------------------------------------------------

* feat: Add project code (`a671a48 <https://github.com/kijanac/luz/commit/a671a48dfdda8eb5a77b040ed4877e8ecb50bfd1>`__)
* feat: First commit (`c32d53a <https://github.com/kijanac/luz/commit/c32d53a1294be0d6a4c1bffa23e44e5ecf7130b0>`__)


`Version 1.0.0 <https://github.com/kijanac/luz/compare/v0.0.1...v1.0.0>`__
--------------------------------------------------------------------------

* feat: Add WrapperDataset,UnpackDataset,OnDiskDataset. Remove DataIterator,GraphDataset,TensorDataset. Temporarily remove IterableDataset,ChainDataset (to be reimplemented later). Lint code. Remove commented code (except for IterableDataset,ChainDataset). Add BaseDataset and make other datasets inherit both this and torch.utils.data.Dataset. Add transform to BaseDataset.loader signature and implement usage accordingly. Fix typo in BaseDataset.loader. Add BaseDataset.random_split. (`8ff3672 <https://github.com/kijanac/luz/commit/8ff3672f435f6d667bced7f4088265bff9932c9c>`__)
* feat: Remove BasicLearner and add functionality to Learner. Add val_dataset to Learner.learn signature (usage to be implemented). Remove extraneous comments and commented code. (`12c6f18 <https://github.com/kijanac/luz/commit/12c6f182af70adfbe005aa642be1dfac774738c6>`__)
* feat: Remove LinearRegressor, NeuralNetwork, Perceptron. Remove transform from Predictor.__init__ signature. Add Predictor.forward. Simplify Predictor.__call__. Add Predictor.eval and rewrite Predictor.predict to use it. Fix Predictor.to type annotation. Add Predictor.train. Remove commented code. (`49d87b2 <https://github.com/kijanac/luz/commit/49d87b24884c335a60c33eed15eb4ea9c784820e>`__)
* feat: Remove extraneous comments and commented code. Rewrite CrossValidationScorer and add fold_seed option for consistent seeded instantiation across folds. Slightly rewrite HoldoutValidationScorer._split_dataset. Lint code (`50fcd49 <https://github.com/kijanac/luz/commit/50fcd492a15fce4cfb7d144238ed9bfaa74fc7f4>`__)
* feat: Remove SupervisedGraphTrainer. Lint code. Clean up Trainer.run. Add val_dataset to Trainer.run signature (usage to be implemented). Remove commented code. Add Trainer.process_batch, Trainer.set_mode, Trainer.migrate, Trainer.backward, and Trainer.optimizer_step. Change dataset.loader params to be passed as kwargs in Trainer.__init__. (`cb5d2f7 <https://github.com/kijanac/luz/commit/cb5d2f71474b00c10c3c5b9634ad3c5c81f757be>`__)
* fix: Remove commented code. Rewrite Transform to operate on luz.Data objects and make all other transforms inherit TensorTransform. Fix incomplete error catching in PowerSeries. (`0a1c331 <https://github.com/kijanac/luz/commit/0a1c331d1a7b91f08c42796507367a51859495a2>`__)
* feat: Remove commented code. Rewrite CrossValidationScorer. Add best_score and best_hyperparameters properties to Tuner. Revamp Tuner.scorer. Add functionality for Tuner to seed each tuning loop for reproducibility. Update type annotations for Tuner. (`98a96b0 <https://github.com/kijanac/luz/commit/98a96b07554b002719a52c2c52154750e2442a94>`__)
* feat: Add set_seed and temporary_seed functions. Remove extraneous comments and commented code. Type annotate memoize and string_to_class. (`bda37ec <https://github.com/kijanac/luz/commit/bda37ec5c27e65695ddfc47d0c0c9b3cdea2f9ec>`__)
* feat: Add project code (`a671a48 <https://github.com/kijanac/luz/commit/a671a48dfdda8eb5a77b040ed4877e8ecb50bfd1>`__)
* feat: First commit (`c32d53a <https://github.com/kijanac/luz/commit/c32d53a1294be0d6a4c1bffa23e44e5ecf7130b0>`__)


`Version 2.0.0 <https://github.com/kijanac/luz/compare/v1.0.0...v2.0.0>`__
--------------------------------------------------------------------------

* feat: Add type annotation to Module. Add GraphNetwork and related commented code. Rewrite WAVE to use Module in __init__ and rewrite WAVE.forward and WAVE._propagate. (`0a0b77c <https://github.com/kijanac/luz/commit/0a0b77c8b754123fa3dfcaaa1057d9ad0c6dd2c9>`__)
* feat: Add default_collate and graph_collate. Refactor dataset classes accordingly and add BaseDataset.use_collate. Add type annotations to BaseDataset.subset and fix bug within. Add type annotations to UnpackDataset. (`2f84ab9 <https://github.com/kijanac/luz/commit/2f84ab91e9bd30a97878f31ea62c7f5d05fd30a3>`__)
* feat: Type annotate Reshape. Add Concatenate module. (`26ccaec <https://github.com/kijanac/luz/commit/26ccaecb18d29fac366d08ab32d564fb4b090d78>`__)
* feat: Update set_seed and temporary_seed to seed Python's random module in addition to numpy and torch (`60034ca <https://github.com/kijanac/luz/commit/60034ca4b36cc98a39e9a8e4f45529c7e100ffd5>`__)


`Version 3.0.0 <https://github.com/kijanac/luz/compare/v2.0.0...v3.0.0>`__
--------------------------------------------------------------------------

* feat: Add batchwise_node_mean, batchwise_node_sum, batchwise_edge_mean, batchwise_edge_sum, nodewise_edge_mean, nodewise_edge_sum, masked_softmax. Lint code. (`c7eb30f <https://github.com/kijanac/luz/commit/c7eb30f9f0015fe3343fe760dc93f60cf072b560>`__)
* feat: Restructure GraphNetwork for more flexibility. Lint code. (`5d80cc4 <https://github.com/kijanac/luz/commit/5d80cc467490bccb5fc3d3452221b0185419c306>`__)


`Version 3.0.1 <https://github.com/kijanac/luz/compare/v3.0.0...v3.0.1>`__
--------------------------------------------------------------------------

* fix: Fix bug in graph_collate: ensure edge_index is long rather than float. (`f4331bb <https://github.com/kijanac/luz/commit/f4331bb09e570a9c9cd3b9be85cd002bbf9fb2d2>`__)


`Version 3.1.0 <https://github.com/kijanac/luz/compare/v3.0.1...v3.1.0>`__
--------------------------------------------------------------------------

* fix: Update graph_collate to collate key 'y' correctly. (`1169fc2 <https://github.com/kijanac/luz/commit/1169fc2c9a09d85a1847cb82d0e72562037916fa>`__)
* feat: Clean up Progress.batch_ended and add Progress.epoch_started to match. Add bar_length arg to Progress.__init__. Add small type annotation to RVP.__init__. (`17603d6 <https://github.com/kijanac/luz/commit/17603d6b60c4a36d4b8143d4bd186e86251400d7>`__)


`Version 3.2.0 <https://github.com/kijanac/luz/compare/v3.1.0...v3.2.0>`__
--------------------------------------------------------------------------

* fix: Fix bug in graph_collate batching of edge_index (`f4bf8eb <https://github.com/kijanac/luz/commit/f4bf8eb6e9563859b617abec593c8c9a1bca3c49>`__)


`Version 3.2.1 <https://github.com/kijanac/luz/compare/v3.2.0...v3.2.1>`__
--------------------------------------------------------------------------

* fix: Fix bug in graph_collate introduced in last commit (`d9b8b2c <https://github.com/kijanac/luz/commit/d9b8b2cf8ee3e5f6cd443c4b67618b35fa40ba37>`__)


`Version 3.3.0 <https://github.com/kijanac/luz/compare/v3.2.1...v3.3.0>`__
--------------------------------------------------------------------------

* fix: Fix bug in BaseDataset.random_split which did not propagate user-supplied collate function to produced subsets. (`be7ad44 <https://github.com/kijanac/luz/commit/be7ad44377e6efb655764eb224fb0b230e65f9c8>`__)
* feat: Add luz.mkdir_safe utility function. (`f5101c8 <https://github.com/kijanac/luz/commit/f5101c820d99d3b889ca1a422b347d84901aa581>`__)
* fix: Fix bug due to incorrect variable reference which caused CrossValidationScorer to produce an error (`a4a5204 <https://github.com/kijanac/luz/commit/a4a5204685337e7d3adb23eb1ccca3777c33f5f2>`__)