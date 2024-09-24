import argparse

from training import Train
from predict import Test

## Create parse argues ##
parser = argparse.ArgumentParser(description='Select between training and categorizing')
parser.add_argument('-t', '--train', action='store_true', help='Train the model on the current training data.')
parser.add_argument('-pathtrain', '--path-train', type=str, help='Path to the folder with the classes to train')
parser.add_argument('-c', '--categorize', action='store_true', help='Categorize unknown data to make new training data')
parser.add_argument('-pathtest', '--path-test', type=str, help='Path to the folder with the classes to train')
parser.add_argument('-n', '--name', type=str, help='Name of the test')
parser.add_argument('-i', '--name-instrument', type=str, help='Name of the instrument: planktivore, issis or microscopy')
parser.add_argument('-v', '--verbose', type=bool, help='DEbugging')

args = parser.parse_args()


# Custom validation logic
if args.train:
    if not args.path_train or not args.name:
        parser.error("--train requires --path-train and --name to be specified.")
    # You can add additional logic here for training with the given path and name

if args.categorize:
    if not args.path_test or not args.name:
        parser.error("--categorize requires --path-test  and --name to be specified.")


if args.train:
    train = Train(args.path_train, args.name, args.name_instrument)
    train.run()
    
if args.test:
    test = Test(args.path_test, args.name, args.name_instrument)
    test.run()

