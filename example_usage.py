#!/usr/bin/env python3
"""
Example usage of the Vitruvian Proportion Analyzer

This script demonstrates different ways to use the analyzer with specific images.
"""

from vitruvius_measurement import analyze_image, VitruvianAnalyzer
import sys

def example_simple_usage():
    """Simple one-line analysis"""
    print("=== SIMPLE USAGE ===")
    try:
        # Analyze with default settings
        results = analyze_image("pose.png")
        print("✓ Analysis completed successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")

def example_custom_outputs():
    """Analysis with custom output paths"""
    print("\n=== CUSTOM OUTPUTS ===")
    try:
        # Analyze with custom output names
        results = analyze_image(
            image_path="pose.png",
            output_path="my_custom_analysis.png",
            save_data="my_data.json",
            show_summary_only=True
        )
        print("✓ Custom analysis completed!")
    except Exception as e:
        print(f"✗ Error: {e}")

def example_programmatic_usage():
    """Advanced programmatic usage"""
    print("\n=== PROGRAMMATIC USAGE ===")
    try:
        # Create analyzer instance for multiple uses
        analyzer = VitruvianAnalyzer()
        
        # Analyze image
        results = analyzer.analyze_proportions("pose.png")
        
        # Get custom analysis
        summary = analyzer.get_analysis_summary(results)
        print("Custom Summary:")
        print(summary)
        
        # Access specific measurements
        head_ratio = results['actual_proportions']['head_to_body']
        ideal_head = results['ideal_proportions']['head_to_body']
        deviation = results['deviations']['head_to_body']['percentage']
        
        print(f"\nHead-to-Body Analysis:")
        print(f"  Measured: {head_ratio:.3f}")
        print(f"  Ideal: {ideal_head:.3f}")
        print(f"  Deviation: {deviation:+.1f}%")
        
        print("✓ Programmatic analysis completed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def example_batch_different_settings():
    """Process the same image with different settings"""
    print("\n=== DIFFERENT SETTINGS ===")
    
    settings = [
        {"confidence": 0.3, "name": "low_confidence"},
        {"confidence": 0.7, "name": "high_confidence"},
    ]
    
    for setting in settings:
        try:
            print(f"Processing with {setting['name']}...")
            results = analyze_image(
                image_path="pose.png",
                output_path=f"{setting['name']}_analysis.png",
                confidence=setting['confidence'],
                show_summary_only=True
            )
            print(f"✓ {setting['name']} completed!")
        except Exception as e:
            print(f"✗ Error with {setting['name']}: {e}")

if __name__ == "__main__":
    print("Vitruvian Proportion Analyzer - Usage Examples")
    print("=" * 50)
    
    # Check if pose.png exists
    import os
    if not os.path.exists("pose.png"):
        print("Error: pose.png not found in current directory.")
        print("Please run this script from the project directory.")
        sys.exit(1)
    
    # Run examples
    example_simple_usage()
    example_custom_outputs() 
    example_programmatic_usage()
    example_batch_different_settings()
    
    print("\n" + "=" * 50)
    print("All examples completed! Check the generated files.")
