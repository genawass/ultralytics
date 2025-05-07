import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Merge COCO vehicle classes into a single car class')
    parser.add_argument('input_json', help='Path to the input COCO JSON file')
    parser.add_argument('output_json', help='Path to save the modified COCO JSON file')
    return parser.parse_args()

# Define vehicle classes that should be merged into 'car'
VEHICLE_CLASSES = {'car', 'truck', 'bus'}

def merge_classes(input_json):
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Create new categories
    new_categories = [
        {'id': 1, 'name': 'car', 'supercategory': 'vehicle'},
        {'id': 0, 'name': 'person', 'supercategory': 'person'}
    ]

    # Create mapping from old category IDs to new ones
    category_mapping = {}
    for cat in data['categories']:
        if cat['name'] in VEHICLE_CLASSES:
            category_mapping[cat['id']] = 1  # Map to car
        elif cat['name'] == 'person':
            category_mapping[cat['id']] = 0  # Map to person
        else:
            continue

    # Update annotations with new category IDs
    new_annotations = []
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id in category_mapping:
            ann['category_id'] = category_mapping[cat_id]
            new_annotations.append(ann)

    # Update the dataset
    data['categories'] = new_categories
    data['annotations'] = new_annotations
    
    return data

def main():
    args = parse_args()
    modified_data = merge_classes(args.input_json)
    
    with open(args.output_json, 'w') as f:
        json.dump(modified_data, f)
    
    print(f"Modified COCO annotations saved to {args.output_json}")

if __name__ == '__main__':
    main()