import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import hamming
import tkinter as tk
from tkinter import simpledialog
from scipy.ndimage import gaussian_filter

def load_and_preprocess(image_path):
    """Load image and enhance for better iris detection"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return img, enhanced

def detect_iris_boundary_from_pupil(image, pupil_center, pupil_radius, display_steps=False):
    """
    Detect iris boundary using radial edge detection from pupil center
    Based on a simplified version of Daugman's integro-differential operator
    """
    height, width = image.shape
    cx, cy = pupil_center
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Enhance edges using a combination of methods
    edges = cv2.Canny(blurred, 20, 100)
    
    # Maximum expected iris radius (can be adjusted)
    max_radius = min(width, height) // 3
    min_radius = pupil_radius * 2  # Iris should be at least twice pupil size
    
    # Safety check
    if min_radius < 10:
        min_radius = 10
    if max_radius > min(width, height) // 2:
        max_radius = min(width, height) // 2
    
    best_radius = None
    best_score = -1
    
    # Storage for visualization
    edge_strength = np.zeros(max_radius + 1)
    
    # Apply integro-differential operator
    for r in range(min_radius, max_radius + 1):
        # Create a circular mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, pupil_center, r, 255, 1)
        
        # Count edge points along the circle
        edge_points = cv2.bitwise_and(edges, mask)
        count = np.sum(edge_points > 0)
        
        # Normalize by circumference
        score = count / (2 * np.pi * r)
        edge_strength[r] = score
        
        if score > best_score:
            best_score = score
            best_radius = r
    
    # Filter the edge strength signal to smooth out noise
    filtered_strength = gaussian_filter(edge_strength, sigma=3)
    
    # Find peaks in the filtered signal
    peaks = []
    for r in range(min_radius + 1, max_radius):
        if (filtered_strength[r] > filtered_strength[r-1] and 
            filtered_strength[r] > filtered_strength[r+1] and
            filtered_strength[r] > 0.1):  # Threshold to filter weak peaks
            peaks.append((r, filtered_strength[r]))
    
    # Sort peaks by strength
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Use the strongest peak as iris radius
    if peaks:
        best_radius = peaks[0][0]
    elif best_radius is None:
        # Fallback if no clear peak is found
        best_radius = int(pupil_radius * 3)  # Typical iris-to-pupil ratio
    
    # Display the edge strength plot if requested
    if display_steps:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(edge_strength)), edge_strength, 'b-', label='Edge Strength')
        plt.plot(range(len(filtered_strength)), filtered_strength, 'r-', label='Filtered Strength')
        if peaks:
            peak_radii = [p[0] for p in peaks]
            peak_strengths = [p[1] for p in peaks]
            plt.plot(peak_radii, peak_strengths, 'go', label='Detected Peaks')
        plt.axvline(x=best_radius, color='m', linestyle='--', label=f'Selected Radius = {best_radius}')
        plt.axvline(x=pupil_radius, color='k', linestyle=':', label=f'Pupil Radius = {pupil_radius}')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Edge Strength')
        plt.title('Iris Boundary Detection')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return best_radius

def robust_iris_localization(image, min_radius=20, max_radius=150, display_steps=True):
    """
    More robust iris localization using a multi-stage approach
    """
    # Create a copy to work on
    img_copy = image.copy()
    height, width = image.shape
    
    # Step 1: Apply bilateral filter to reduce noise but preserve edges
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Step 2: Create a histogram of pixel intensities and enhance contrast
    hist_eq = cv2.equalizeHist(image)
    
    # Step 3: Apply Gaussian blur for further noise reduction
    blurred = cv2.GaussianBlur(hist_eq, (5, 5), 0)
    
    # Step 4: Detect edges using Canny
    edges = cv2.Canny(blurred, 30, 80)
    
    # Step 5: Try to detect pupil first (usually the darkest region)
    # First threshold to isolate dark regions
    _, pupil_thresh1 = cv2.threshold(bilateral, 40, 255, cv2.THRESH_BINARY_INV)
    
    # Second threshold with Otsu's method as backup
    _, pupil_thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine thresholds
    pupil_thresh = cv2.bitwise_and(pupil_thresh1, pupil_thresh2)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_OPEN, kernel)
    pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find pupil contours
    contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    pupil_center = None
    pupil_radius = None
    iris_center = None
    iris_radius = None
    
    # Try to find the pupil from contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # Skip tiny contours
            continue
            
        # Fit an ellipse to better handle non-circular pupils
        try:
            (x, y), (width_ellipse, height_ellipse), angle = cv2.fitEllipse(contour)
            # Use average of width/height as radius
            radius = int((width_ellipse + height_ellipse) / 4)  # Divide by 4 to get radius from diameters
            center = (int(x), int(y))
            
            # Check if the ellipse is reasonably centered in the image
            if (x > width * 0.2 and x < width * 0.8 and 
                y > height * 0.2 and y < height * 0.8 and
                radius >= 5 and radius <= min_radius * 2):
                pupil_center = center
                pupil_radius = radius
                break
        except:
            # Fallback to enclosing circle if ellipse fitting fails
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Check if the circle is reasonably centered in the image
            if (x > width * 0.2 and x < width * 0.8 and 
                y > height * 0.2 and y < height * 0.8 and
                radius >= 5 and radius <= min_radius * 2):
                pupil_center = center
                pupil_radius = radius
                break
    
    # Step 6: If pupil is found, use it to find iris boundary
    if pupil_center is not None:
        # Use specialized iris boundary detection
        iris_radius = detect_iris_boundary_from_pupil(image, pupil_center, pupil_radius, display_steps)
        iris_center = pupil_center
    
    # Step 7: If pupil detection failed, try Hough transform for both
    if pupil_center is None:
        # Try different parameter combinations
        param2_values = [30, 25, 20, 15, 10]  # Accumulator threshold
        
        # Try to find both pupil and iris
        found_circles = False
        
        for param2 in param2_values:
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1,
                minDist=width/8,
                param1=50,
                param2=param2,
                minRadius=5,
                maxRadius=max_radius
            )
            
            if circles is not None:
                # Sort circles by radius
                circles = np.uint16(np.around(circles[0, :]))
                circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
                
                if len(circles) >= 2:
                    # Smallest circle is likely pupil, largest is likely iris
                    pupil_circle = circles[0]
                    iris_circle = circles[-1]
                    
                    # Check if circles are concentric (approximately)
                    pupil_center = (int(pupil_circle[0]), int(pupil_circle[1]))
                    pupil_radius = int(pupil_circle[2])
                    
                    iris_center = (int(iris_circle[0]), int(iris_circle[1]))
                    iris_radius = int(iris_circle[2])
                    
                    # Check if these are reasonable values
                    distance_between_centers = np.sqrt(
                        (pupil_center[0] - iris_center[0])**2 + 
                        (pupil_center[1] - iris_center[1])**2
                    )
                    
                    # Centers should be close and iris should be larger than pupil
                    if distance_between_centers <= pupil_radius and iris_radius > pupil_radius * 1.5:
                        found_circles = True
                        break
                elif len(circles) == 1:
                    # If only one circle is found, assume it's the iris
                    iris_circle = circles[0]
                    iris_center = (int(iris_circle[0]), int(iris_circle[1]))
                    iris_radius = int(iris_circle[2])
                    
                    # Estimate pupil from iris
                    pupil_center = iris_center
                    pupil_radius = int(iris_radius / 3)  # Typical ratio
                    found_circles = True
                    break
        
        if not found_circles:
            # None of the automated methods worked
            if display_steps:
                print("Automated detection failed.")
    
    # Step 8: If all automatic detection failed, allow manual input
    if iris_center is None or pupil_center is None:
        if display_steps:
            print("Automatic iris/pupil detection failed. You will need to specify manually.")
            plt.figure(figsize=(10, 8))
            plt.imshow(image, cmap='gray')
            plt.title("Original Image - Automatic Detection Failed")
            plt.show()
            
            # Use tkinter for manual input
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Get iris parameters
            iris_x = simpledialog.askinteger("Manual Input", "Enter iris center X coordinate:", 
                                          initialvalue=width//2, minvalue=0, maxvalue=width)
            iris_y = simpledialog.askinteger("Manual Input", "Enter iris center Y coordinate:", 
                                          initialvalue=height//2, minvalue=0, maxvalue=height)
            iris_r = simpledialog.askinteger("Manual Input", "Enter iris radius:", 
                                          initialvalue=min(width, height)//6, minvalue=10, maxvalue=min(width, height)//2)
                                          
            # Get pupil parameters
            pupil_x = simpledialog.askinteger("Manual Input", "Enter pupil center X coordinate:", 
                                           initialvalue=iris_x, minvalue=0, maxvalue=width)
            pupil_y = simpledialog.askinteger("Manual Input", "Enter pupil center Y coordinate:", 
                                           initialvalue=iris_y, minvalue=0, maxvalue=height)
            pupil_r = simpledialog.askinteger("Manual Input", "Enter pupil radius:", 
                                           initialvalue=iris_r//3, minvalue=1, maxvalue=iris_r)
            
            iris_center = (iris_x, iris_y)
            iris_radius = iris_r
            pupil_center = (pupil_x, pupil_y)
            pupil_radius = pupil_r
        else:
            # Use image center as fallback if not displaying steps
            iris_center = (width // 2, height // 2)
            iris_radius = min(width, height) // 4
            pupil_center = iris_center
            pupil_radius = iris_radius // 3
    
    # Display intermediate steps if requested
    if display_steps:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        
        plt.subplot(2, 3, 2)
        plt.imshow(hist_eq, cmap='gray')
        plt.title("Histogram Equalization")
        
        plt.subplot(2, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection")
        
        plt.subplot(2, 3, 4)
        plt.imshow(pupil_thresh, cmap='gray')
        plt.title("Pupil Threshold")
        
        plt.subplot(2, 3, 5)
        detected = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        if iris_center and iris_radius:
            cv2.circle(detected, iris_center, iris_radius, (0, 255, 0), 2)
        if pupil_center and pupil_radius:
            cv2.circle(detected, pupil_center, pupil_radius, (0, 0, 255), 2)
        plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        plt.title("Detection Result")
        
        # Create zoomed in view
        plt.subplot(2, 3, 6)
        zoom_factor = 2  # Adjust as needed
        zoom_size = int(iris_radius * zoom_factor)
        
        # Create zoomed region
        zoom_x_start = max(0, iris_center[0] - zoom_size)
        zoom_x_end = min(width, iris_center[0] + zoom_size)
        zoom_y_start = max(0, iris_center[1] - zoom_size)
        zoom_y_end = min(height, iris_center[1] + zoom_size)
        
        zoomed = detected[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
        plt.imshow(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
        plt.title("Zoomed Detection")
        
        plt.tight_layout()
        plt.show()
    
    return iris_center, iris_radius, pupil_center, pupil_radius

def normalize_iris(gray, iris_center, iris_radius, pupil_center, pupil_radius):
    """Convert iris from circular to rectangular form (rubber sheet model)"""
    height = 64  # Height of normalized iris image
    width = 512  # Width of normalized iris image
    
    normalized = np.zeros((height, width), dtype=np.uint8)
    
    # Extract center coordinates
    cx_iris, cy_iris = iris_center
    cx_pupil, cy_pupil = pupil_center
    
    # Calculate polar coordinates
    for y in range(height):
        theta = 2.0 * np.pi * y / height
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for x in range(width):
            # Map to position between pupil and iris boundary
            r_ratio = x / width
            
            # Account for non-concentric pupil
            r_pupil = pupil_radius
            r_iris = iris_radius
            
            # Interpolate between pupil and iris
            r = (1 - r_ratio) * r_pupil + r_ratio * r_iris
            
            # Calculate coordinates in the original image
            xi = cx_pupil + r * cos_theta + r_ratio * (cx_iris - cx_pupil)
            yi = cy_pupil + r * sin_theta + r_ratio * (cy_iris - cy_pupil)
            
            # Check bounds and copy pixel
            xi_int, yi_int = int(round(xi)), int(round(yi))
            if 0 <= xi_int < gray.shape[1] and 0 <= yi_int < gray.shape[0]:
                normalized[y, x] = gray[yi_int, xi_int]
    
    return normalized

def extract_features(normalized_iris):
    """Extract features using Local Binary Patterns"""
    radius = 3
    n_points = 8 * radius
    
    # Apply LBP to get texture features
    lbp = local_binary_pattern(normalized_iris, n_points, radius, method='uniform')
    
    # Create binary code (simplified IrisCode)
    threshold = np.mean(lbp)
    binary_code = (lbp > threshold).astype(np.uint8)
    
    return binary_code

def calculate_hamming_distance(code1, code2):
    """Calculate Hamming distance between two binary codes"""
    # Flatten the codes
    code1_flat = code1.flatten()
    code2_flat = code2.flatten()
    
    # Calculate Hamming distance
    distance = hamming(code1_flat, code2_flat)
    return distance

def find_best_match(code1, code2):
    """Find best match with rotational alignment"""
    min_distance = 1.0
    best_rotation = 0
    
    # Try different rotational alignments
    rotations = 8  # Number of rotations to try
    for r in range(rotations):
        rotation_amount = r * (code1.shape[0] // rotations)
        rotated_code = np.roll(code2, rotation_amount, axis=0)
        
        distance = calculate_hamming_distance(code1, rotated_code)
        if distance < min_distance:
            min_distance = distance
            best_rotation = rotation_amount
    
    return min_distance, best_rotation

def recognize_iris(img1_path, img2_path, display_steps=True):
    """Complete iris recognition process comparing two eye images"""
    # Process first image
    img1, enhanced1 = load_and_preprocess(img1_path)
    iris_center1, iris_radius1, pupil_center1, pupil_radius1 = robust_iris_localization(enhanced1, display_steps=display_steps)
    normalized1 = normalize_iris(enhanced1, iris_center1, iris_radius1, pupil_center1, pupil_radius1)
    code1 = extract_features(normalized1)
    
    # Process second image
    img2, enhanced2 = load_and_preprocess(img2_path)
    iris_center2, iris_radius2, pupil_center2, pupil_radius2 = robust_iris_localization(enhanced2, display_steps=display_steps)
    normalized2 = normalize_iris(enhanced2, iris_center2, iris_radius2, pupil_center2, pupil_radius2)
    code2 = extract_features(normalized2)
    
    # Find best match with rotational alignment
    distance, best_rotation = find_best_match(code1, code2)
    
    # Apply the best rotation for visualization
    best_rotated_normalized = np.roll(normalized2, best_rotation, axis=0)
    best_rotated_code = np.roll(code2, best_rotation, axis=0)
    
    # Visualize final results
    plt.figure(figsize=(15, 10))
    
    # First image with detection
    plt.subplot(2, 3, 1)
    img1_with_circles = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2RGB)
    cv2.circle(img1_with_circles, iris_center1, iris_radius1, (0, 255, 0), 2)
    cv2.circle(img1_with_circles, pupil_center1, pupil_radius1, (255, 0, 0), 2)
    plt.imshow(img1_with_circles)
    plt.title('Image 1 Detection')
    
    # Normalized iris 1
    plt.subplot(2, 3, 2)
    plt.imshow(normalized1, cmap='gray')
    plt.title('Normalized Iris 1')
    
    # IrisCode 1
    plt.subplot(2, 3, 3)
    plt.imshow(code1, cmap='binary')
    plt.title('IrisCode 1')
    
    # Second image with detection
    plt.subplot(2, 3, 4)
    img2_with_circles = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2RGB)
    cv2.circle(img2_with_circles, iris_center2, iris_radius2, (0, 255, 0), 2)
    cv2.circle(img2_with_circles, pupil_center2, pupil_radius2, (255, 0, 0), 2)
    plt.imshow(img2_with_circles)
    plt.title('Image 2 Detection')
    
    # Normalized iris 2
    plt.subplot(2, 3, 5)
    plt.imshow(best_rotated_normalized, cmap='gray')
    plt.title(f'Normalized Iris 2 (rot: {best_rotation})')
    
    # IrisCode 2
    plt.subplot(2, 3, 6)
    plt.imshow(best_rotated_code, cmap='binary')
    plt.title('IrisCode 2')
    
    plt.suptitle(f'Hamming Distance: {distance:.4f}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Interpretation of results
    print(f"Hamming Distance: {distance:.4f}")
    if distance < 0.32:
        print("SAME IRIS (high confidence)")
    elif distance < 0.36:
        print("LIKELY SAME IRIS (medium confidence)")
    else:
        print("DIFFERENT IRISES")
    
    return distance

if __name__ == "__main__":
    try:
        # Replace with your image paths
        distance = recognize_iris("same_eyes/S5001L00.jpg", "same_eyes/S5001L09.jpg")
        print(f"Final Hamming distance: {distance:.4f}")
        print("For reference: The Afghan Girl had Hamming Distances of 0.24 (left eye) and 0.31 (right eye).")
    except Exception as e:
        print(f"Error: {e}")