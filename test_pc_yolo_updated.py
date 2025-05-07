import torch
from ultralytics import YOLO
import traceback
import os

def test_model_summary(model_path):
    """Load model and print its summary"""
    try:
        # Check if file exists
        if not os.path.exists(model_path) and not model_path.startswith("yolo"): # Allow built-in models like yolov8s.pt
            print(f"Error: Model configuration file not found: {model_path}")
            return False

        print(f'Loading model: {model_path}')
        model = YOLO(model_path)
        print('Model successfully loaded!')

        # Print model info (includes layers, parameters, GFLOPs)
        model.info(verbose=True) # Remove imgsz=640 argument

        # Create dummy input for model verification
        dummy_input = torch.zeros(1, 3, 640, 640) # Standard input size
        print(f'Dummy input shape: {dummy_input.shape}')

        # Test forward pass
        print('Running forward pass...')
        output = model.model(dummy_input)
        print('Model forward pass successful!')

        # Print detection head output shapes
        if isinstance(output, list):
            # For models with multiple output heads (like detection models)
            for i, out_tensor in enumerate(output):
                if isinstance(out_tensor, torch.Tensor):
                    print(f'Output head {i} shape: {out_tensor.shape}')
                else:
                    # Handle cases where output might be a tuple (e.g., RTDETR)
                    print(f'Output head {i} type: {type(out_tensor)}')
                    if isinstance(out_tensor, tuple):
                        for j, t_item in enumerate(out_tensor):
                             if isinstance(t_item, torch.Tensor):
                                print(f'  Output head {i}, item {j} shape: {t_item.shape}')
        elif isinstance(output, torch.Tensor):
            # For models with a single output tensor (like classification models)
            print(f'Output shape: {output.shape}')
        else:
            print(f"Unexpected output type: {type(output)}")


        # Count total parameters (already in model.info(), but can be explicitly calculated)
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

        return True
    except Exception as e:
        print(f'Error testing model {model_path}: {e}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    pc_yolo_model_path = 'ultralytics/cfg/models/11/pc-yolo11s_.yaml'

    print(f"\n=== Testing Updated PC-YOLO (P2,P3,P4 Head) ===")
    success = test_model_summary(pc_yolo_model_path)

    if success:
        print(f"\nPC-YOLO model at {pc_yolo_model_path} tested successfully.")
    else:
        print(f"\nPC-YOLO model at {pc_yolo_model_path} test failed.")

    # As a comparison, let's also test a standard model if it's relevant
    # print("\n=== Testing Standard YOLOv8s for comparison ===")
    # standard_success = test_model_summary('yolov8s.yaml') # yolov8s.yaml is usually available
    # if standard_success:
    #     print("\nStandard YOLOv8s model tested successfully.")
    # else:
    #     print("\nStandard YOLOv8s model test failed.") 