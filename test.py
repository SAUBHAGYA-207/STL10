import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from model import get_stl_resnet18
import argparse
import os

def test():
    # Rule 2.7.2: Dataset directory must be a configurable input [cite: 72, 93]
    parser = argparse.ArgumentParser(description='STL-10 Test Evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to STL-10 dataset')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to model.pth')
    args, unknown = parser.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Architecture from model.py [cite: 81, 88]
    model = get_stl_resnet18().to(device)
    
    # Load Weights [cite: 76]
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found.")
        return

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set to FP16 and Eval mode for consistency with train.py saving
    model.half().eval()

    # Normalization constants used during training
    norm = transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    
    # Load Test Set [cite: 28, 94]
    testset = torchvision.datasets.STL10(root=args.data_dir, split='test', download=True)
    
    correct = 0
    total = len(testset)
    
    print(f"🚀 Starting 10-View TTA Evaluation on {total} images...")

    with torch.no_grad():
        for i in range(total):
            img, target = testset[i]
            
            # 10-View TTA: 5 Crops + 5 Horizontal Flips of those crops
            # Using 88x88 crops resized back to 96x96 to maintain detail
            crops = transforms.functional.five_crop(img, 88) 
            views = []
            for c in crops:
                resized = transforms.functional.resize(c, (96, 96))
                # Add original crop
                views.append(norm(transforms.ToTensor()(resized).half()))
                # Add flipped crop
                views.append(norm(transforms.ToTensor()(transforms.functional.hflip(resized)).half()))
            
            # Create a batch of 10 views for a single image
            batch = torch.stack(views).to(device)
            
            # Get predictions and average the probabilities (Ensemble effect)
            outputs = model(batch)
            avg_probs = F.softmax(outputs.float(), dim=1).mean(0)
            
            if avg_probs.argmax().item() == target:
                correct += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Evaluated {i + 1}/{total} images...")

    # Rule 6.2.3: Final classification accuracy computation [cite: 94]
    final_acc = 100. * correct / total
    print("\n" + "="*40)
    print(f"🌟 FINAL TEST ACCURACY: {final_acc:.2f}%")
    print("="*40)
    
    if final_acc >= 85.0:
        print("✅ Status: QUALIFIED for Accuracy Points")
    else:
        print("❌ Status: BELOW 85% THRESHOLD (Zero Accuracy Points)")

if __name__ == '__main__':
    test()