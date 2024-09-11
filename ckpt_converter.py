import torch

def convert_state_dict(state_dict):
    new_state_dict = {}
    
    # Convert 'model' prefix to 'net' prefix
    for key, value in state_dict.items():
        # Remove the initial "model."
        if key.startswith("model."):
            new_key = key.replace("model.", "net.", 1)  # Replace only the first occurrence
        else:
            new_key = key

        # Update the new state dict
        new_state_dict[new_key] = value
    
    return new_state_dict

def load_and_convert_checkpoint(input_ckpt_path, output_ckpt_path):
    # Load the checkpoint
    checkpoint = torch.load(input_ckpt_path)
    
    # Convert the state dict
    converted_state_dict = convert_state_dict(checkpoint['state_dict'])
    
    # Replace the original state dict with the converted one
    checkpoint['state_dict'] = converted_state_dict
    
    # Save the modified checkpoint
    torch.save(checkpoint, output_ckpt_path)
    print(f"Converted checkpoint saved to: {output_ckpt_path}")

if __name__ == "__main__":
    input_ckpt_path = "/workspace/vrd/ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/best.ckpt"
    output_ckpt_path = "/workspace/vrd/ckpt/egtr__pretrained_detr__SenseTime__deformable-detr__batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune__version_0/batch__64__epochs__50_25__lr__2e-07_2e-06_0.0002__visual_genome__finetune/version_0/checkpoints/modified.ckpt"
    
    load_and_convert_checkpoint(input_ckpt_path, output_ckpt_path)
