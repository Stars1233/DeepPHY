import cv2
import numpy as np

def detect_bubbles(image_path, output_image_path):
    """
    Detects bubbles in an image using pre-optimized OpenCV parameters,
    saves an annotated image showing the detected bubbles, and returns their
    relative coordinates. This function is designed for automation and does not
    display any UI windows.

    Args:
        image_path (str): The path to the image file to process.
        output_image_path (str): The path to save the annotated image.

    Returns:
        dict: A dictionary where keys are bubble IDs (as strings, e.g., "1", "2")
              and values are dictionaries containing the relative 'x' and 'y'
              coordinates. Returns an empty dictionary if no bubbles are found
              or the image cannot be loaded.
              Example: {"1": {"x": 0.5, "y": 0.3}, "2": {"x": 0.8, "y": 0.4}}
    """
    # These are the pre-tuned parameters for bubble detection.
    params = {
        'threshold_value': 218,
        'dp': 1.1,
        'minDist': 101,
        'param1': 255,
        'param2': 37,
        'minRadius': 20,
        'maxRadius': 60
    }

    image = cv2.imread(image_path)
    if image is None:
        print(f"[Bubble Detector] Error: Cannot load image at {image_path}")
        return {}

    # Create a copy for drawing annotations
    annotated_image = image.copy()
    img_h, img_w = image.shape[:2]

    # Image preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, preprocessed_image = cv2.threshold(gray, params['threshold_value'], 255, cv2.THRESH_BINARY)

    # Use Hough Circle Transform to find circles
    circles = cv2.HoughCircles(preprocessed_image,
                               cv2.HOUGH_GRADIENT,
                               dp=params['dp'],
                               minDist=params['minDist'],
                               param1=params['param1'],
                               param2=params['param2'],
                               minRadius=params['minRadius'],
                               maxRadius=params['maxRadius'])

    detected_bubbles_coords = {}
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sort circles by their x-coordinate to ensure consistent ID assignment
        sorted_circles = sorted(circles, key=lambda c: c[0])

        for i, (x, y, r) in enumerate(sorted_circles):
            # The LLM is prompted with 1-based IDs, so we generate them here.
            bubble_id = str(i + 1)
            # Convert pixel coordinates to relative coordinates
            rel_x = x / img_w
            rel_y = y / img_h
            detected_bubbles_coords[bubble_id] = {'x': rel_x, 'y': rel_y}

            # --- Annotation Drawing ---
            color = (0, 0, 255)  # Red in BGR
            thickness = 4
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8

            # Draw the circle boundary
            cv2.circle(annotated_image, (x, y), r, color, thickness)
            # Draw the center point
            cv2.circle(annotated_image, (x, y), 5, color, -1)  # -1 for filled circle

            # Prepare and draw the label
            label = f"Bubble {bubble_id}"
            # Position the text below the circle
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x - text_width // 2
            text_y = y + r + 25  # Offset below the circle's bottom edge
            cv2.putText(annotated_image, label, (text_x, text_y), font, font_scale, color, thickness)

        # Save the annotated image
        cv2.imwrite(output_image_path, annotated_image)
        print(f"  - [Bubble Detector] Saved annotated bubble image to: {output_image_path}")

    return detected_bubbles_coords
