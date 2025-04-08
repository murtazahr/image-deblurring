import torch
import numpy as np
from PIL import Image
import argparse
import os
import sys

try:
    from model import Generator
except ImportError as e:
    print(f"Error importing Generator: {e}")
    print("Make sure you're running this script from the same directory as model.py")
    sys.exit(1)


def normalization(images):
    """Normalize images to [-1, 1] range"""
    return (images - 127.5) / 127.5


def denormalization(images):
    """Denormalize images from [-1, 1] range to [0, 255]"""
    return (images * 127.5 + 127.5).clip(0, 255).astype(np.uint8)


def process_image(input_path, output_path, model_path, image_size=None):
    """
    Load a trained generator model and process a single image.

    Args:
        input_path: Path to the input blurry image
        output_path: Path to save the enhanced image
        model_path: Path to the trained generator model weights
        image_size: Optional tuple (height, width) to resize the image
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist")
        return False

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist")
        return False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    print(f"Loading model from {model_path}")
    try:
        generator = Generator().to(device)

        # Load with map_location and weights_only=True for safer loading
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint)
        generator.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load and preprocess the image
    print(f"Loading image from {input_path}")
    try:
        image = Image.open(input_path)
        image = image.convert('RGB')  # Ensure RGB format
        original_size = image.size
        print(f"Image loaded, original size: {original_size}")

        # Resize image if specified
        if image_size:
            print(f"Resizing image to {image_size}")
            image = image.resize(image_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        image_np = np.array(image).astype(np.float32)
        print(f"Image array shape: {image_np.shape}")

        # Normalize using the same function as in training
        image_np = normalization(image_np)

        # Convert to PyTorch tensor and adjust channels (HWC to CHW)
        image_tensor = torch.FloatTensor(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
        print(f"Input tensor shape: {image_tensor.shape}")
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Process the image
    print("Processing image through generator")
    try:
        with torch.no_grad():
            enhanced_image = generator(image_tensor)
        print(f"Processing complete, output tensor shape: {enhanced_image.shape}")
    except Exception as e:
        print(f"Error during model inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Convert back to numpy and adjust range to [0, 255]
    print("Converting output to image")
    try:
        enhanced_np = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_np = denormalization(enhanced_np)

        # Save the result
        enhanced_pil = Image.fromarray(enhanced_np)

        # Resize back to original size if it was resized
        if image_size and original_size != image_size:
            enhanced_pil = enhanced_pil.resize(original_size, Image.Resampling.LANCZOS)

        enhanced_pil.save(output_path)
        print(f"Enhanced image saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving output image: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Process an image using the trained generator model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="enhanced_output.png", help="Path to save output image")
    parser.add_argument("--model", type=str, default="weight/generator.pth", help="Path to model weights")
    parser.add_argument("--size", type=str, help="Optional size to resize image (format: WIDTHxHEIGHT)")
    args = parser.parse_args()

    # Parse size if provided
    image_size = None
    if args.size:
        try:
            width, height = map(int, args.size.lower().split('x'))
            image_size = (width, height)
            print(f"Will resize image to {width}x{height}")
        except:
            print(f"Invalid size format: {args.size}. Should be WIDTHxHEIGHT (e.g., 256x256)")
            return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process the image
    success = process_image(args.input, args.output, args.model, image_size)

    if not success:
        print("Image processing failed")
        sys.exit(1)
    else:
        print("Image processing completed successfully")


if __name__ == "__main__":
    main()
