import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

def load_and_preprocess_images(ref_path, input_path):
    """
    Load and preprocess reference and input images
    """
    # Load images
    ref_img = plt.imread(ref_path)
    input_img = plt.imread(input_path)
    
    # Convert to grayscale if needed
    if len(ref_img.shape) == 3:
        ref_img = np.mean(ref_img, axis=2)
    if len(input_img.shape) == 3:
        input_img = np.mean(input_img, axis=2)
    
    return ref_img, input_img

def resize_images(ref_img, input_img):
    """
    Resize images to the same dimensions
    """
    # Get minimum dimensions
    min_height = min(ref_img.shape[0], input_img.shape[0])
    min_width = min(ref_img.shape[1], input_img.shape[1])
    
    # Resize images
    ref_resized = ref_img[:min_height, :min_width]
    input_resized = input_img[:min_height, :min_width]
    
    return ref_resized, input_resized

def calculate_similarity_metrics(ref_img, input_img):
    """
    Calculate similarity metrics between two images
    """
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(ref_img - input_img))
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((ref_img - input_img) ** 2)
    
    return mae, mse

def compare_faces(ref_path, input_path, threshold=0.1):
    """
    Main function to compare two face images
    """
    # Load and preprocess images
    ref_img, input_img = load_and_preprocess_images(ref_path, input_path)
    
    # Resize to same dimensions
    ref_resized, input_resized = resize_images(ref_img, input_img)
    
    # Calculate similarity metrics
    mae, mse = calculate_similarity_metrics(ref_resized, input_resized)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Reference image
    axes[0].imshow(ref_resized, cmap='gray')
    axes[0].set_title('Reference Face')
    axes[0].axis('off')
    
    # Input image
    axes[1].imshow(input_resized, cmap='gray')
    axes[1].set_title('Input Face')
    axes[1].axis('off')
    
    # Difference image
    diff_img = np.abs(ref_resized - input_resized)
    axes[2].imshow(diff_img, cmap='hot')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    # Determine match result
    if mse < threshold:
        result = "Face Matched"
        color = 'green'
    else:
        result = "Face Not Matched"
        color = 'red'
    
    # Add result text
    plt.figtext(0.5, 0.01, 
                f'Result: {result} | MSE: {mse:.4f} | MAE: {mae:.4f}', 
                ha='center', fontsize=12, color=color, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return mse, mae, result

def batch_comparison(ref_path, input_folder, threshold=0.1):
    """
    Compare reference image with multiple input images
    """
    results = []
    
    # Load reference image
    ref_img, _ = load_and_preprocess_images(ref_path, ref_path)
    
    # Process each input image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            
            # Load and preprocess input image
            input_img, _ = load_and_preprocess_images(input_path, input_path)
            
            # Resize to match reference
            ref_resized, input_resized = resize_images(ref_img, input_img)
            
            # Calculate metrics
            mae, mse = calculate_similarity_metrics(ref_resized, input_resized)
            
            # Determine result
            match_result = "Matched" if mse < threshold else "Not Matched"
            
            results.append({
                'Image': filename,
                'MSE': mse,
                'MAE': mae,
                'Result': match_result
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display results
    print("Comparison Results:")
    print(df)
    
    # Plot MSE values
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Image'], df['MSE'], color=['green' if x < threshold else 'red' for x in df['MSE']])
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('MSE Values for Face Comparison')
    plt.xlabel('Input Images')
    plt.ylabel('MSE Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df

# Example usage
if __name__ == "__main__":
    # Single comparison
    print("Single Face Comparison:")
    face_reference = r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW04\p4\face_reference.png'
    # face_input = r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW04\p4\face_input1.png'
    face_input = r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW04\p4\face_input2.png'
    
    # Check if files exist
    print(f"Reference file exists: {os.path.exists(face_reference)}")
    print(f"Input file exists: {os.path.exists(face_input)}")
    
    if os.path.exists(face_reference) and os.path.exists(face_input):
        mse, mae, result = compare_faces(face_reference, face_input)
        print(f"\n MSE: {mse}, MAE: {mae}, Result: {result}")
    else:
        print("\n One or both files do not exist!")

    # Batch comparison (if you have multiple images)
    # print("\nBatch Face Comparison:")
    # df_results = batch_comparison('face_reference.png', 'input_images_folder/')
    
    # Analysis and conclusions
    print("\n--- Analysis and Conclusions ---")
    print("1. How much difference is there between images?")
    print(f"   - MSE: {mse:.4f}, MAE: {mae:.4f}")
    print("   - Lower values indicate higher similarity")
    
    print("\n2. Does our simple system perform well in face recognition?")
    print("   - This is a very basic system using pixel-wise comparison")
    print("   - It works for identical/similar images but lacks robustness")
    print("   - Not suitable for real-world face recognition applications")
    
    print("\n3. What factors might cause errors in recognition?")
    print("   - Lighting conditions (brightness, contrast differences)")
    print("   - Face angle and rotation")
    print("   - Facial expressions")
    print("   - Image resolution and quality")
    print("   - Background variations")
    print("   - Scale and size differences")
    print("   - Occlusions (glasses, masks, hair)")
    