import argparse
import os
import json

def parse_args():
	desc = "Creates a PyTorch-compatible dataset.json file based on folders" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	args = parser.parse_args()
	return args


def main():
	global args
	global labels
	labels = {"labels": []}
	args = parse_args()

	# If the input folder is a directory
	if os.path.isdir(args.input_folder):
		print("Processing folder: " + args.input_folder)
	else:
		print("Not a working input_folder path: " + args.input_folder)
		return

	# Get the subdirectories
	for root, subdirs, files in os.walk(args.input_folder):
		subs = subdirs
		break
	# Iterate through the subdirectories and show the classification "key"
	for index, subdir in enumerate(subs):
		print("Loading images from " + subdir + " as label " + str(index))
	# Loop through the dataset and create the data structure needed for the JSON
	for root, subdirs, files in os.walk(args.input_folder):
		for filename in files:
			# Skip .DS_Store files and the like
			if filename.startswith("."):
				continue
			path = os.path.join(root, filename).split(args.input_folder)[1].strip(os.sep)
			labels["labels"].append([path, subs.index(path.split(os.sep)[0])])	

	# Write out the dataset.json file
	with open(os.path.join(args.input_folder, "dataset.json"), "w") as outfile:
		json.dump(labels, outfile)


if __name__ == "__main__":
	main()