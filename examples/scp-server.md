# SCP Commands for Server Data Transfer

Transfer data from remote server to local machine.

## Server Info

- Host: `10.123.4.30`
- User: `sjia`
- Remote Path: `/home/sjia/projects/Reasoning-Autoregressive-Modeling/`

## Examples

### Download logs

```bash
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/uTEST-ed_train/logs ./EXPERIMENT/uTEST-ed_train/logs
```

### Download checkpoints

```bash
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/uTEST-ed_train/checkpoints ./EXPERIMENT/uTEST-ed_train/checkpoints
```

### Download training history

```bash
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/uTEST-ed_train/logs/training_history.json ./EXPERIMENT/uTEST-ed_train/logs/
```

### Download entire experiment folder

```bash
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/uTEST-ed_train ./EXPERIMENT/
```

### Download C3 training outputs

```bash
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/c3_original/logs ./EXPERIMENT/c3_original/logs
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/c3_original/checkpoints ./EXPERIMENT/c3_original/checkpoints
```

### Download PreExp-c3_original logs

- all
```bash
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/PreExp-c3_original/checkpoints/checkpoint_final.pt ./EXPERIMENT/PreExp-c3_original/checkpoints/
```

- logs
```bash
scp -r sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/PreExp-c3_original/logs ./EXPERIMENT/PreExp-c3_original/logs
```

## Upload to Server

### Upload config changes

```bash
scp ./configs/PreExp/c3_original.yml sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/configs/PreExp/
```

### Upload code changes

```bash
scp ./examples/PreExp/c3_original.py sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/examples/PreExp/
scp ./ram/models/encoder.py sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/ram/models/
scp ./ram/models/decoder.py sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/ram/models/
```

## Sync with rsync

For large folders, use rsync instead of scp:

```bash
rsync -avz --progress sjia@10.123.4.30:/home/sjia/projects/Reasoning-Autoregressive-Modeling/EXPERIMENT/ ./EXPERIMENT/
```
