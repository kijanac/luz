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
* fix: Fix bug in graph_collate introduced in last commit (`21737cb <https://github.com/kijanac/luz/commit/21737cbf9397f1429685ddb4f5f59f7f7a8bc669>`__)
* fix: Fix bug in graph_collate batching of edge_index (`367f55b <https://github.com/kijanac/luz/commit/367f55bd0122d59002fce2f5623917b0711a64f8>`__)
* fix: Update graph_collate to collate key 'y' correctly. (`9219eba <https://github.com/kijanac/luz/commit/9219ebaba9d79fe411a3a5f4c44453756b21b172>`__)
* feat: Clean up Progress.batch_ended and add Progress.epoch_started to match. Add bar_length arg to Progress.__init__. Add small type annotation to RVP.__init__. (`e7d0879 <https://github.com/kijanac/luz/commit/e7d08790372022d9a9065f7f709d271e35fe3a3f>`__)
* fix: Fix bug in graph_collate: ensure edge_index is long rather than float. (`6e168c5 <https://github.com/kijanac/luz/commit/6e168c5ffde15f1aeaea082354b0060a9e483da1>`__)
* feat: Add batchwise_node_mean, batchwise_node_sum, batchwise_edge_mean, batchwise_edge_sum, nodewise_edge_mean, nodewise_edge_sum, masked_softmax. Lint code. (`1579ab3 <https://github.com/kijanac/luz/commit/1579ab3d4a9d72011e18432ff9ee4d8178a3b33c>`__)
* feat: Restructure GraphNetwork for more flexibility. Lint code. (`905e219 <https://github.com/kijanac/luz/commit/905e2196cfececb308eb05f356883a1ce85fa1f4>`__)
* feat: Add type annotation to Module. Add GraphNetwork and related commented code. Rewrite WAVE to use Module in __init__ and rewrite WAVE.forward and WAVE._propagate. (`cffff6e <https://github.com/kijanac/luz/commit/cffff6ee04e818b3c6aab42d06d2be07b2182306>`__)
* feat: Add default_collate and graph_collate. Refactor dataset classes accordingly and add BaseDataset.use_collate. Add type annotations to BaseDataset.subset and fix bug within. Add type annotations to UnpackDataset. (`080312d <https://github.com/kijanac/luz/commit/080312dce099c4077f9ecf7d1b1b1ae4a0c26d99>`__)
* feat: Type annotate Reshape. Add Concatenate module. (`7df1c5f <https://github.com/kijanac/luz/commit/7df1c5f019fef6546fa1d6a2f9b8e7e0a8a5d589>`__)
* feat: Update set_seed and temporary_seed to seed Python's random module in addition to numpy and torch (`957bab8 <https://github.com/kijanac/luz/commit/957bab8420c2ed71f42316d18be86b2da84c563a>`__)
* feat: Add WrapperDataset,UnpackDataset,OnDiskDataset. Remove DataIterator,GraphDataset,TensorDataset. Temporarily remove IterableDataset,ChainDataset (to be reimplemented later). Lint code. Remove commented code (except for IterableDataset,ChainDataset). Add BaseDataset and make other datasets inherit both this and torch.utils.data.Dataset. Add transform to BaseDataset.loader signature and implement usage accordingly. Fix typo in BaseDataset.loader. Add BaseDataset.random_split. (`ea4debc <https://github.com/kijanac/luz/commit/ea4debce86694400e3ec0718e385de5918d98c8a>`__)
* feat: Remove BasicLearner and add functionality to Learner. Add val_dataset to Learner.learn signature (usage to be implemented). Remove extraneous comments and commented code. (`44593c0 <https://github.com/kijanac/luz/commit/44593c09eaee20e310be1b6fc14478410ca01a7f>`__)
* feat: Remove LinearRegressor, NeuralNetwork, Perceptron. Remove transform from Predictor.__init__ signature. Add Predictor.forward. Simplify Predictor.__call__. Add Predictor.eval and rewrite Predictor.predict to use it. Fix Predictor.to type annotation. Add Predictor.train. Remove commented code. (`f5fd92a <https://github.com/kijanac/luz/commit/f5fd92ad4545cbcf6f0c357d0dc8cb93e0abc6e6>`__)
* feat: Remove extraneous comments and commented code. Rewrite CrossValidationScorer and add fold_seed option for consistent seeded instantiation across folds. Slightly rewrite HoldoutValidationScorer._split_dataset. Lint code (`71c80d5 <https://github.com/kijanac/luz/commit/71c80d5e43b937f07b550926f3a1ae6ec6aa9971>`__)
* feat: Remove SupervisedGraphTrainer. Lint code. Clean up Trainer.run. Add val_dataset to Trainer.run signature (usage to be implemented). Remove commented code. Add Trainer.process_batch, Trainer.set_mode, Trainer.migrate, Trainer.backward, and Trainer.optimizer_step. Change dataset.loader params to be passed as kwargs in Trainer.__init__. (`6a93edc <https://github.com/kijanac/luz/commit/6a93edc18029fb5610abcf13dad2ec3fbcc6e440>`__)
* fix: Remove commented code. Rewrite Transform to operate on luz.Data objects and make all other transforms inherit TensorTransform. Fix incomplete error catching in PowerSeries. (`f3116d5 <https://github.com/kijanac/luz/commit/f3116d56cc52c8ed1dd7405e1339a82fc2a87961>`__)
* feat: Remove commented code. Rewrite CrossValidationScorer. Add best_score and best_hyperparameters properties to Tuner. Revamp Tuner.scorer. Add functionality for Tuner to seed each tuning loop for reproducibility. Update type annotations for Tuner. (`8ded4e1 <https://github.com/kijanac/luz/commit/8ded4e1c56080729a5ecf841f47208db274bc046>`__)
* feat: Add set_seed and temporary_seed functions. Remove extraneous comments and commented code. Type annotate memoize and string_to_class. (`d31c649 <https://github.com/kijanac/luz/commit/d31c64934cc9044535b6cd8d9065fb402d22371c>`__)
* feat: Add project code (`a671a48 <https://github.com/kijanac/luz/commit/a671a48dfdda8eb5a77b040ed4877e8ecb50bfd1>`__)
* feat: First commit (`c32d53a <https://github.com/kijanac/luz/commit/c32d53a1294be0d6a4c1bffa23e44e5ecf7130b0>`__)


`Version 4.0.0 <https://github.com/kijanac/luz/compare/v3.3.0...v4.0.0>`__
--------------------------------------------------------------------------

* feat: Add attention, batchwise_mask, and nodewise_mask utility functions. Remove extraneous utility functions evaluate_expression and string_to_class. (`7fb6993 <https://github.com/kijanac/luz/commit/7fb6993d0a2f16d8788892206ae1f604696f4a3e>`__)
* feat: Rename FC and FCRNN to Dense and DenseRNN, respectively. Add EdgeAttention, MultiheadEdgeAttention, Squeeze, and Unsqueeze modules. (`5caf7bd <https://github.com/kijanac/luz/commit/5caf7bdf9d293b5da1fb671dc7330efcb3bb6c41>`__)


`Version 5.0.0 <https://github.com/kijanac/luz/compare/v4.0.0...v5.0.0>`__
--------------------------------------------------------------------------

* feat: Add Standardize, Center, CenterPerTensor transforms. (`5ab3647 <https://github.com/kijanac/luz/commit/5ab364721651489b767f7f22d4d7a74028caf3a6>`__)
* feat: Add functionality to save and load Trainer state. (`ef4423a <https://github.com/kijanac/luz/commit/ef4423a6ce7c18993831d372bc44e7ffb15964ea>`__)
* feat: Greatly expand Module functionality and modularity. (`b02ee08 <https://github.com/kijanac/luz/commit/b02ee0800d39eddd96a1623ea98fcf1fa20b915c>`__)
* feat: Convert Learner to an AbstractBaseClass (`b55cbc2 <https://github.com/kijanac/luz/commit/b55cbc28df59d9cab91446c16339242cce3a03ab>`__)
* fix: Fix bug which prevented use of BaseDataset.Subset (`036b120 <https://github.com/kijanac/luz/commit/036b12032f3b0136e58bf05079b37a9e4c640684>`__)
* feat: Add VALIDATING flag for use in training handlers. (`7245363 <https://github.com/kijanac/luz/commit/7245363a4885cfc66ab573cbc4213dd40991496b>`__)
* feat: Remove Predictor and move functionality to Module. (`a59bd4c <https://github.com/kijanac/luz/commit/a59bd4c1e992eb3ce66639bcd3fca80c1ae7121b>`__)
* feat: Add PlotHistory handler. (`30cb8d1 <https://github.com/kijanac/luz/commit/30cb8d12f9c6f736b70663dacd93cefd42d7c65b>`__)
* feat: Add ActualVsPredicted handler. (`3a29d6b <https://github.com/kijanac/luz/commit/3a29d6b3d9029a26f25208cc42d465343b9a4a39>`__)
* feat: Enable CUDA functionality. (`2bb8741 <https://github.com/kijanac/luz/commit/2bb8741b8a3bdcbcd974ffb509db121065b1b4e7>`__)
* feat: Add Accuracy and FBeta handlers. Remove superfluous Validate handler. (`f2bfd96 <https://github.com/kijanac/luz/commit/f2bfd9645cb87e64ad1056a30ef8d86ace8d93e8>`__)
* feat: Add early stopping capabilities to Trainer. (`281fda4 <https://github.com/kijanac/luz/commit/281fda4b5559ea7c606ecc8682a0e607b1cbc79a>`__)
* feat: Add Validate handler. (`0e91c33 <https://github.com/kijanac/luz/commit/0e91c33ad380c3dc722baa46a5544352f00aeca7>`__)
* feat: Implement validation dataset in Trainer.run. (`07402c5 <https://github.com/kijanac/luz/commit/07402c52498e5a5f9944f4b2c899c57945476a6a>`__)
* fix: Fix bug which prevented Trainer from skipping gradient calculations in test mode. (`2690368 <https://github.com/kijanac/luz/commit/2690368cfcf65b49a41820707f2dd1ccc56e6603>`__)
* feat: Update Predictor.eval to include no_grad option. (`2368e04 <https://github.com/kijanac/luz/commit/2368e046a66d34dfa9934e016460a333eff65214>`__)
* feat: Rename attention utility function to dot_product_attention. (`233900b <https://github.com/kijanac/luz/commit/233900b662b3ff3bccee2cff4ea9b6695aa13840>`__)
* feat: Add AdditiveNodeAttention, DotProductAttention, and MaskedSoftmax modules. Remove WAVE module. (`cb0bd2b <https://github.com/kijanac/luz/commit/cb0bd2b2212bfb0f3f15748b9f74172743ac07b1>`__)
* feat: Add AdditiveAttention and GraphConvAttention modules. (`14153ad <https://github.com/kijanac/luz/commit/14153adc7a468940d265b9bf9a0049259b61b6a8>`__)
* feat: Add GraphConv module. (`d649f07 <https://github.com/kijanac/luz/commit/d649f0743699960471be8b4568a9ee6472fd1401>`__)
* feat: Add adjacency, in_degree, out_degree utility functions for graph manipulations. (`ad3f3b6 <https://github.com/kijanac/luz/commit/ad3f3b624c26f0db2976ace794f1f5dd1622e3b7>`__)


`Version 6.0.0 <https://github.com/kijanac/luz/compare/v5.0.0...v6.0.0>`__
--------------------------------------------------------------------------

* feat: Add functionality for Trainer to log messages to a file through Trainer.log and Trainer.log_filepath. (`4d0aec2 <https://github.com/kijanac/luz/commit/4d0aec215658d2623b648173eb17e1035c7bd847>`__)
* feat: Add DurbinWatson handler. Add options to show plot and save plot to PlotHistory. Replace print statements in all handlers with model.log (`37bd946 <https://github.com/kijanac/luz/commit/37bd9467f925e9b8667a21fed50f1130ef15a02d>`__)
* feat: Add aggregate_mask and remove_self_loops functions (`16844bb <https://github.com/kijanac/luz/commit/16844bb47b159991aa562a371cf7b827c519369b>`__)
* feat: Add AverageGraphPool module. Modify Module.log to refer to logging messages rather than updating Trainer state. (`63d4cc2 <https://github.com/kijanac/luz/commit/63d4cc260adfa2fb85096a981d04e6d5690de5b0>`__)
* feat: Update tuners to conform with new Learner interface (`93b141e <https://github.com/kijanac/luz/commit/93b141e4a902c29ec2501f9e9a24b0a49e46c53c>`__)
* feat: Add new features to Learner to reduce boilerplate code and simplify scoring and tuning (`cc879e5 <https://github.com/kijanac/luz/commit/cc879e5a0586800afe87fb1af290cd2fc25e96ac>`__)


`Version 6.0.0 <https://github.com/kijanac/luz/compare/v6.0.0...v6.0.0>`__
--------------------------------------------------------------------------


