#!/usr/bin/env python3
"""
Batch processing example for Vitruvian Proportion Analyzer.

This script demonstrates how to process multiple images and generate
a comparative analysis report.
"""

import os
import json
from pathlib import Path
from vitruvius_measurement import VitruvianAnalyzer


def batch_analyze(input_dir: str, output_dir: str):
    """
    Analyze all images in a directory and save results.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = VitruvianAnalyzer()
    
    # Supported image extensions
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process")
    
    results_summary = []
    
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        
        try:
            # Analyze image
            results = analyzer.analyze_proportions(str(image_path))
            
            # Save individual analysis
            output_name = image_path.stem
            json_path = Path(output_dir) / f"{output_name}_analysis.json"
            vis_path = Path(output_dir) / f"{output_name}_visualization.png"
            
            analyzer.save_analysis_data(results, str(json_path))
            analyzer.visualize_analysis(str(image_path), results, str(vis_path))
            
            # Add to summary
            summary_data = {
                'filename': image_path.name,
                'summary': analyzer.get_analysis_summary(results),
                'deviations': results['deviations']
            }
            results_summary.append(summary_data)
            
            print(f"  ✓ Processed successfully")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")
            continue
    
    # Save batch summary
    summary_path = Path(output_dir) / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nBatch processing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process images for Vitruvian analysis")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        exit(1)
    
    batch_analyze(args.input_dir, args.output_dir)
