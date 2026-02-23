---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:958
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- map
- mrr@10
- ndcg@10
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-reranking
      name: Cross Encoder Reranking
    dataset:
      name: wiki eval
      type: wiki-eval
    metrics:
    - type: map
      value: 1.0
      name: Map
    - type: mrr@10
      value: 1.0
      name: Mrr@10
    - type: ndcg@10
      value: 1.0
      name: Ndcg@10
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['What is r respect safety constraints during the learning and/or deployment processes?', 'r respect safety constraints during the learning and/or deployment processes. An alternative approach is risk-averse reinforcement learning, where instead of the expected return, a risk-measure of the return is optimized, such as the conditional value at risk (CVaR). In addition to mitigating risk, the CVaR objective increases robustness to model uncertainties. However, CVaR optimization in risk-averse RL requires special care, to prevent gradient bias and blindness to success.\n\n\n=== Self-reinfo'],
    ['What is ment learning is where an agent take actions in an environment to maximize the accumulation of rewards?', 'is to solve the problem of mode collapse (see above). The authors claim "In no experiment did we see evidence of mode collapse for the WGAN algorithm".\n\n\n=== GANs with more than two players ===\n\n\n==== Adversarial autoencoder ====\nAn adversarial autoencoder (AAE) is more autoencoder than GAN. The idea is to start with a plain autoencoder, but train a discriminator to discriminate the latent vectors from a reference distribution (often the normal distribution).\n\n\n==== InfoGAN ====\nIn conditional'],
    ['What is n the grand prize in 2009 for $1 million?', 'n the Grand Prize in 2009 for $1 million. Shortly after the prize was awarded, Netflix realised that viewers\' ratings were not the best indicators of their viewing patterns ("everything is a recommendation") and they changed their recommendation engine accordingly. In 2010, an article in The Wall Street Journal noted the use of machine learning by Rebellion Research to predict the 2008 financial crisis. In 2012, co-founder of Sun Microsystems, Vinod Khosla, predicted that 80% of medical doctors'],
    ['What is e instantiated in computational models, making them predecessors of deep learning systems?', 'e instantiated in computational models, making them predecessors of deep learning systems. These developmental models share the property that various proposed learning dynamics in the brain (e.g., a wave of nerve growth factor) support the self-organization somewhat analogous to the neural networks utilized in deep learning models. Like the neocortex, neural networks employ a hierarchy of layered filters in which each layer considers information from a prior layer (or the operating environment),'],
    ['What is ard and convergent training algorithms?', 'work grow uncontrollably large—but this is managed with shortcuts called skip connections in residual networks. Another theory is that batch normalization adjusts data by handling its size and path separately, speeding up training.\n\n\n== Internal covariate shift ==\nEach layer in a neural network has inputs that follow a specific distribution, which shifts during training due to two main factors: the random starting values of the network’s settings (parameter initialization) and the natural variat'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'What is r respect safety constraints during the learning and/or deployment processes?',
    [
        'r respect safety constraints during the learning and/or deployment processes. An alternative approach is risk-averse reinforcement learning, where instead of the expected return, a risk-measure of the return is optimized, such as the conditional value at risk (CVaR). In addition to mitigating risk, the CVaR objective increases robustness to model uncertainties. However, CVaR optimization in risk-averse RL requires special care, to prevent gradient bias and blindness to success.\n\n\n=== Self-reinfo',
        'is to solve the problem of mode collapse (see above). The authors claim "In no experiment did we see evidence of mode collapse for the WGAN algorithm".\n\n\n=== GANs with more than two players ===\n\n\n==== Adversarial autoencoder ====\nAn adversarial autoencoder (AAE) is more autoencoder than GAN. The idea is to start with a plain autoencoder, but train a discriminator to discriminate the latent vectors from a reference distribution (often the normal distribution).\n\n\n==== InfoGAN ====\nIn conditional',
        'n the Grand Prize in 2009 for $1 million. Shortly after the prize was awarded, Netflix realised that viewers\' ratings were not the best indicators of their viewing patterns ("everything is a recommendation") and they changed their recommendation engine accordingly. In 2010, an article in The Wall Street Journal noted the use of machine learning by Rebellion Research to predict the 2008 financial crisis. In 2012, co-founder of Sun Microsystems, Vinod Khosla, predicted that 80% of medical doctors',
        'e instantiated in computational models, making them predecessors of deep learning systems. These developmental models share the property that various proposed learning dynamics in the brain (e.g., a wave of nerve growth factor) support the self-organization somewhat analogous to the neural networks utilized in deep learning models. Like the neocortex, neural networks employ a hierarchy of layered filters in which each layer considers information from a prior layer (or the operating environment),',
        'work grow uncontrollably large—but this is managed with shortcuts called skip connections in residual networks. Another theory is that batch normalization adjusts data by handling its size and path separately, speeding up training.\n\n\n== Internal covariate shift ==\nEach layer in a neural network has inputs that follow a specific distribution, which shifts during training due to two main factors: the random starting values of the network’s settings (parameter initialization) and the natural variat',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Reranking

* Dataset: `wiki-eval`
* Evaluated with [<code>CERerankingEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator) with these parameters:
  ```json
  {
      "at_k": 10
  }
  ```

| Metric      | Value   |
|:------------|:--------|
| map         | 1.0     |
| mrr@10      | 1.0     |
| **ndcg@10** | **1.0** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 958 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 958 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                        | label                                                         |
  |:--------|:-------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                            | float                                                         |
  | details | <ul><li>min: 34 characters</li><li>mean: 149.49 characters</li><li>max: 509 characters</li></ul> | <ul><li>min: 172 characters</li><li>mean: 487.25 characters</li><li>max: 500 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>What is r respect safety constraints during the learning and/or deployment processes?</code>                           | <code>r respect safety constraints during the learning and/or deployment processes. An alternative approach is risk-averse reinforcement learning, where instead of the expected return, a risk-measure of the return is optimized, such as the conditional value at risk (CVaR). In addition to mitigating risk, the CVaR objective increases robustness to model uncertainties. However, CVaR optimization in risk-averse RL requires special care, to prevent gradient bias and blindness to success.<br><br><br>=== Self-reinfo</code>                       | <code>1.0</code> |
  | <code>What is ment learning is where an agent take actions in an environment to maximize the accumulation of rewards?</code> | <code>is to solve the problem of mode collapse (see above). The authors claim "In no experiment did we see evidence of mode collapse for the WGAN algorithm".<br><br><br>=== GANs with more than two players ===<br><br><br>==== Adversarial autoencoder ====<br>An adversarial autoencoder (AAE) is more autoencoder than GAN. The idea is to start with a plain autoencoder, but train a discriminator to discriminate the latent vectors from a reference distribution (often the normal distribution).<br><br><br>==== InfoGAN ====<br>In conditional</code> | <code>0.0</code> |
  | <code>What is n the grand prize in 2009 for $1 million?</code>                                                               | <code>n the Grand Prize in 2009 for $1 million. Shortly after the prize was awarded, Netflix realised that viewers' ratings were not the best indicators of their viewing patterns ("everything is a recommendation") and they changed their recommendation engine accordingly. In 2010, an article in The Wall Street Journal noted the use of machine learning by Rebellion Research to predict the 2008 financial crisis. In 2012, co-founder of Sun Microsystems, Vinod Khosla, predicted that 80% of medical doctors</code>                                 | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | wiki-eval_ndcg@10 |
|:-----:|:----:|:-----------------:|
| 1.0   | 60   | 1.0               |
| 2.0   | 120  | 1.0               |
| 3.0   | 180  | 1.0               |


### Framework Versions
- Python: 3.10.0
- Sentence Transformers: 5.2.3
- Transformers: 5.2.0
- PyTorch: 2.10.0+cpu
- Accelerate: 1.12.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->