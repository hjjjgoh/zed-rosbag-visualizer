import torch

# Load checkpoint
checkpoint = torch.load("models/model_best_bp2.pth", map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Check if there's an 'args' field
if 'args' in checkpoint:
    print("\nStored args:")
    print(checkpoint['args'])
elif 'config' in checkpoint:
    print("\nStored config:")
    print(checkpoint['config'])

# Check a specific layer shape to understand the architecture
print("\nSample layer shapes:")
for key, value in list(checkpoint['model'].items())[:10]:
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")
    
# Check the problematic layer
problematic_key = 'update_block.encoder.convc1.weight'
if problematic_key in checkpoint['model']:
    print(f"\nProblematic layer '{problematic_key}' shape: {checkpoint['model'][problematic_key].shape}")
