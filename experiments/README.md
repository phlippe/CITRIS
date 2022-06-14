# Experiment run-files

This folder contains all files to start an experiment with the implemented methods.
Note that for any experiment, you first need to have generated a dataset.
See the files in the folder `data_generation/` for further information on the dataset generation.

## Autoencoder

To train an autoencoder, you can use the following command:
```bash
python train_ae.py --data_dir ../data/pinball/ --num_latents 24
```

## Causal Encoder

To train a causal encoder, you can use the following command:
```bash
python train_causal_encoder.py --data_dir ../data/pinball/
```

## (i)CITRIS-VAE

To train a iCITRIS-VAE, you can use the following command:
```bash
python train_vae.py --model iCITRISVAE \
                    --data_dir ../data/voronoi_6vars_random_seed42/ \
                    --causal_encoder_checkpoint ../data/voronoi_6vars_random_seed42/models/CausalEncoder.ckpt \
                    --num_latents 12 \
                    --beta_t1 10 \
                    --beta_classifier 10 \
                    --beta_mi_estimator 10 \
                    --graph_learning_method ENCO \
                    --lambda_sparse 0.02
```
For running NOTEARS, use `--graph_learning_method NOTEARS --lambda_sparse 0.002` instead.

For running CITRIS instead of iCITRIS, use `--model CITRISVAE`.

## (i)CITRIS-NF

To train a iCITRIS-NF, you can use the following command:
```bash
python train_nf.py --model iCITRISVAE \
                   --data_dir ../data/pinball/ \
                   --causal_encoder_checkpoint ../data/pinball/models/CausalEncoder.ckpt \
                   --autoencoder_checkpoint ../data/pinball/models/Autoencoder.ckpt \
                   --num_latents 24 \
                   --beta_classifier 4 \
                   --beta_mi_estimator 4 \
                   --graph_learning_method ENCO \
                   --lambda_sparse 0.02
```

For running CITRIS instead of iCITRIS, use `--model CITRISNF`.