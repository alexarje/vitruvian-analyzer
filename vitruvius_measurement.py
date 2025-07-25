"""
Vitruvian Proportion Analyzer

A Python tool that analyzes human body proportions against the classical 
Vitruvian ideal using computer vision and pose estimation.

Author: Vitruvian Proportion Analyzer Project
License: MIT
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import argparse
import sys
import os

class VitruvianAnalyzer:
    """
    A class for analyzing human body proportions against Vitruvian ideals.
    
    This analyzer uses MediaPipe pose estimation to detect human body landmarks
    and compares the detected proportions against classical Vitruvian standards.
    """
    
    def __init__(self):
        """Initialize the VitruvianAnalyzer with MediaPipe pose estimation."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Vitruvian ideal proportions
        self.ideal_proportions = {
            'head_to_body': 1/8,  # Head height should be 1/8 of total height
            'face_to_body': 1/10,  # Face height should be 1/10 of total height
            'navel_ratio': 5/8,   # Navel should be 5/8 from feet
            'foot_to_body': 1/6,  # Foot length should be 1/6 of height
        }
    
    def get_landmark_coords(self, landmarks, landmark_id: int) -> Tuple[float, float]:
        """Extract x, y coordinates for a specific landmark."""
        landmark = landmarks.landmark[landmark_id]
        return (landmark.x, landmark.y)
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def analyze_proportions(self, image_path: str) -> Dict:
        """Analyze body proportions from an image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Process image with MediaPipe
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            raise ValueError("No pose detected in the image")
        
        landmarks = results.pose_landmarks
        
        # Extract key body landmarks (in normalized coordinates)
        nose = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.NOSE)
        left_eye = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_EYE)
        right_eye = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_EYE)
        left_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_ankle = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        left_heel = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HEEL)
        right_heel = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HEEL)
        left_foot_index = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        right_foot_index = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
        
        # Calculate key measurements
        # Head top (improved approximation)
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        # Use a more conservative multiplier for head top estimation
        head_top_y = eye_center_y - (nose[1] - eye_center_y) * 1.2  # Improved approximation
        
        # Body measurements
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                          (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                     (left_hip[1] + right_hip[1]) / 2)
        ankle_center = ((left_ankle[0] + right_ankle[0]) / 2, 
                       (left_ankle[1] + right_ankle[1]) / 2)
        
        # Total body height (head top to feet)
        total_height = ankle_center[1] - head_top_y
        
        # Head height (head top to chin/neck)
        head_height = shoulder_center[1] - head_top_y
        
        # Face height (approximate: eyes to chin)
        face_height = shoulder_center[1] - eye_center_y
        
        # Navel height (approximate: hip center)
        navel_height_from_feet = ankle_center[1] - hip_center[1]
        
        # Foot length (heel to toe) - use a more conservative approach
        # Note: MediaPipe's FOOT_INDEX might not be at the very tip of the foot
        left_foot_length = self.calculate_distance(left_heel, left_foot_index)
        right_foot_length = self.calculate_distance(right_heel, right_foot_index)
        avg_foot_length = (left_foot_length + right_foot_length) / 2
        
        # Apply a correction factor since FOOT_INDEX is not the actual toe tip
        # This is an approximation based on typical foot anatomy
        avg_foot_length *= 1.15  # Foot is typically 15% longer than heel to foot_index
        
        # Calculate actual proportions
        actual_proportions = {
            'head_to_body': head_height / total_height,
            'face_to_body': face_height / total_height,
            'navel_ratio': navel_height_from_feet / total_height,
            'foot_to_body': avg_foot_length / total_height,
        }
        
        # Calculate deviations from ideal
        deviations = {}
        for key in self.ideal_proportions:
            deviation = actual_proportions[key] - self.ideal_proportions[key]
            deviation_percent = (deviation / self.ideal_proportions[key]) * 100
            deviations[key] = {
                'absolute': deviation,
                'percentage': deviation_percent
            }
        
        return {
            'actual_proportions': actual_proportions,
            'ideal_proportions': self.ideal_proportions,
            'deviations': deviations,
            'landmarks': landmarks,
            'image_shape': (height, width)
        }
    
    def visualize_analysis(self, image_path: str, analysis_results: Dict, save_path: Optional[str] = None):
        """Visualize the analysis results."""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Draw pose landmarks on image
        annotated_image = image_rgb.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image, 
            analysis_results['landmarks'], 
            self.mp_pose.POSE_CONNECTIONS
        )
        
        ax1.imshow(annotated_image)
        ax1.set_title("Pose Detection")
        ax1.axis('off')
        
        # Create proportion comparison chart
        proportions = list(self.ideal_proportions.keys())
        ideal_values = [self.ideal_proportions[p] for p in proportions]
        actual_values = [analysis_results['actual_proportions'][p] for p in proportions]
        
        x = np.arange(len(proportions))
        width = 0.35
        
        ax2.bar(x - width/2, ideal_values, width, label='Vitruvian Ideal', alpha=0.8)
        ax2.bar(x + width/2, actual_values, width, label='Actual', alpha=0.8)
        
        ax2.set_xlabel('Body Proportions')
        ax2.set_ylabel('Ratio')
        ax2.set_title('Proportion Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.replace('_', ' ').title() for p in proportions], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_analysis(self, analysis_results: Dict):
        """Print detailed analysis results."""
        print("VITRUVIAN PROPORTION ANALYSIS")
        print("=" * 40)
        
        for prop_name in self.ideal_proportions:
            ideal = self.ideal_proportions[prop_name]
            actual = analysis_results['actual_proportions'][prop_name]
            deviation = analysis_results['deviations'][prop_name]
            
            print(f"\n{prop_name.replace('_', ' ').title()}:")
            print(f"  Ideal:      {ideal:.3f}")
            print(f"  Actual:     {actual:.3f}")
            print(f"  Deviation:  {deviation['percentage']:+.1f}%")
        
        # Add debug information about measurements
        print(f"\nDEBUG INFO:")
        print(f"Image shape: {analysis_results['image_shape']}")
        print("Note: Large foot measurement deviations may indicate:")
        print("- MediaPipe's foot landmarks don't reach the actual toe tip")
        print("- The pose may not show the full foot clearly")
        print("- The person may be wearing shoes that obscure foot landmarks")
    
    def save_analysis_data(self, analysis_results: Dict, output_path: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Path to save the JSON file
        """
        import json
        
        # Convert to JSON-serializable format
        json_data = {
            'actual_proportions': analysis_results['actual_proportions'],
            'ideal_proportions': analysis_results['ideal_proportions'],
            'deviations': analysis_results['deviations'],
            'image_shape': analysis_results['image_shape']
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Analysis data saved to: {output_path}")
    
    def get_analysis_summary(self, analysis_results: Dict) -> str:
        """
        Get a brief summary of the analysis results.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            String summary of the analysis
        """
        deviations = analysis_results['deviations']
        avg_deviation = np.mean([abs(dev['percentage']) for dev in deviations.values()])
        
        summary = f"Average deviation from Vitruvian ideal: {avg_deviation:.1f}%\n"
        
        # Find the proportion closest to ideal
        closest_prop = min(deviations.keys(), 
                          key=lambda k: abs(deviations[k]['percentage']))
        summary += f"Closest to ideal: {closest_prop.replace('_', ' ').title()} "
        summary += f"({deviations[closest_prop]['percentage']:+.1f}%)\n"
        
        # Find the proportion furthest from ideal
        furthest_prop = max(deviations.keys(), 
                           key=lambda k: abs(deviations[k]['percentage']))
        summary += f"Furthest from ideal: {furthest_prop.replace('_', ' ').title()} "
        summary += f"({deviations[furthest_prop]['percentage']:+.1f}%)"
        
        return summary


def analyze_image(image_path: str, 
                 output_path: Optional[str] = None,
                 save_data: Optional[str] = None,
                 show_summary_only: bool = False,
                 confidence: float = 0.5) -> Dict:
    """
    Simple function to analyze a single image with default settings.
    
    Args:
        image_path: Path to the input image
        output_path: Path for visualization output (auto-generated if None)
        save_data: Path for JSON data output (optional)
        show_summary_only: If True, only print summary instead of full analysis
        confidence: Detection confidence threshold (0.0-1.0)
        
    Returns:
        Dictionary containing analysis results
        
    Example:
        >>> results = analyze_image("my_photo.jpg")
        >>> results = analyze_image("photo.jpg", "analysis.png", "data.json")
    """
    # Validate input
    if not validate_image_file(image_path):
        raise ValueError(f"Invalid image file: {image_path}")
    
    # Set up output paths
    if output_path is None:
        output_path, _ = auto_generate_output_names(image_path)
    
    # Create analyzer
    analyzer = VitruvianAnalyzer()
    if confidence != 0.5:
        analyzer.pose = analyzer.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=confidence
        )
    
    # Analyze image
    results = analyzer.analyze_proportions(image_path)
    
    # Print results
    if show_summary_only:
        print("ANALYSIS SUMMARY")
        print("=" * 20)
        print(analyzer.get_analysis_summary(results))
    else:
        analyzer.print_analysis(results)
    
    # Save data if requested
    if save_data:
        analyzer.save_analysis_data(results, save_data)
        print(f"Analysis data saved to: {save_data}")
    
    # Create visualization
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    analyzer.visualize_analysis(image_path, results, output_path)
    print(f"Visualization saved to: {output_path}")
    
    return results

def validate_image_file(image_path: str) -> bool:
    """
    Validate if the image file exists and has a supported format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        return False
    
    # Check file extension
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    file_ext = os.path.splitext(image_path)[1].lower()
    
    return file_ext in supported_extensions


def auto_generate_output_names(input_image: str) -> Tuple[str, str]:
    """
    Generate automatic output file names based on input image.
    
    Args:
        input_image: Path to input image
        
    Returns:
        Tuple of (visualization_path, data_path)
    """
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    vis_name = f"{base_name}_vitruvian_analysis.png"
    data_name = f"{base_name}_analysis_data.json"
    
    return vis_name, data_name


def main():
    """
    Main function with command-line argument support.
    
    Supports both default example usage and custom image analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze human body proportions against Vitruvian ideals",
        epilog="Example: python vitruvius_measurement.py --image my_photo.jpg --output analysis.png"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="pose.png",
        help="Path to input image (default: pose.png)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path for output visualization (default: auto-generated based on input)"
    )
    parser.add_argument(
        "--auto-output",
        action="store_true",
        help="Automatically generate output filenames based on input image name"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the visualization (useful for batch processing)"
    )
    parser.add_argument(
        "--save-data",
        type=str,
        help="Save analysis data to JSON file (default: auto-generated if --auto-output used)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only summary instead of detailed analysis"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported image formats and exit"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_formats:
        print("Supported image formats:")
        print("- JPEG (.jpg, .jpeg)")
        print("- PNG (.png)")
        print("- BMP (.bmp)")
        print("- TIFF (.tiff, .tif)")
        print("- WebP (.webp)")
        sys.exit(0)
    
    # Validate and set up file paths
    if not validate_image_file(args.image):
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
        else:
            print(f"Error: '{args.image}' is not a supported image format.")
            print("Use --list-formats to see supported formats.")
        sys.exit(1)
    
    # Auto-generate output names if requested or no output specified
    if args.auto_output or not args.output:
        auto_vis, auto_data = auto_generate_output_names(args.image)
        output_path = args.output if args.output else auto_vis
        data_path = args.save_data if args.save_data else (auto_data if args.auto_output else None)
    else:
        output_path = args.output
        data_path = args.save_data
    
    analyzer = VitruvianAnalyzer()
    
    # Update confidence threshold if specified
    if args.confidence != 0.5:
        analyzer.pose = analyzer.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=args.confidence
        )
    
    try:
        print(f"Analyzing image: {args.image}")
        print("-" * 50)
        
        # Analyze proportions
        results = analyzer.analyze_proportions(args.image)
        
        # Print results
        if args.summary_only:
            print("ANALYSIS SUMMARY")
            print("=" * 20)
            print(analyzer.get_analysis_summary(results))
        else:
            analyzer.print_analysis(results)
        
        # Save data if requested
        if data_path:
            analyzer.save_analysis_data(results, data_path)
        
        # Create visualization
        if args.no_display:
            # Temporarily disable matplotlib display
            import matplotlib
            matplotlib.use('Agg')
        
        analyzer.visualize_analysis(args.image, results, output_path)
        
        print(f"\nVisualization saved to: {output_path}")
        if data_path:
            print(f"Analysis data saved to: {data_path}")
        
    except ValueError as e:
        print(f"Analysis Error: {e}")
        print("\nTips for better results:")
        print("- Ensure the image contains a clearly visible human figure")
        print("- Use images with good lighting and contrast")
        print("- Make sure the person is standing upright")
        print("- Include the full body from head to feet")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your input image and try again.")
        sys.exit(1)


def example_usage():
    """Example usage of the VitruvianAnalyzer (legacy function)."""
    analyzer = VitruvianAnalyzer()
    
    # Use the pose.png image in the current directory
    image_path = "pose.png"
    
    try:
        # Analyze proportions
        results = analyzer.analyze_proportions(image_path)
        
        # Print results
        analyzer.print_analysis(results)
        
        # Visualize results
        analyzer.visualize_analysis(image_path, results, "vitruvian_analysis.png")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the image contains a clearly visible human figure.")

if __name__ == "__main__":
    main()