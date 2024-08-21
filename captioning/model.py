import os
import folder_paths

def snapshot_download(repo_id: str, exDir: str=''):

    model_checkpoint = os.path.join(folder_paths.models_dir, exDir, os.path.basename(repo_id))

    if not os.path.exists(model_checkpoint):

        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)

    return model_checkpoint

def hf_hub_download(repo_id: str, filename: str, prefix: str='', repo_type: str="spaces"):
    
    model_checkpoint = os.path.join(folder_paths.models_dir, prefix, filename)

    if not os.path.exists(model_checkpoint):

        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=repo_id, local_dir=model_checkpoint, repo_type=repo_type, filename=filename)

    return model_checkpoint