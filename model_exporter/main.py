import torch
from safetensors import torch as stt

# Load the state_dict
state_dict = torch.load("../models/best.pt", map_location=torch.device("cpu"))

# Extract the model from the state_dict
if "model" in state_dict and hasattr(state_dict["model"], "state_dict"):
    model_state_dict = state_dict["model"].state_dict()  # Extract the model weights
else:
    raise ValueError("The model key is missing or does not have a state_dict method.")

# Save the extracted weights to a safetensors file
print(model_state_dict.keys())
safetensors_path = "model.safetensors"
stt.save_file(model_state_dict, safetensors_path)
print(f"Model weights exported to {safetensors_path}")
