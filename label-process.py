#!/bin/python3

import yaml
import sys

# Finds all unique classes in the given file
# output can be 'print' or 'return'
def find_all_classes(labels_file, output):
  data = yaml.safe_load(open(labels_file, 'r'))
  labels = set()

  # scan data for labels
  for entry in data:
    for box in entry['boxes']:
      labels.add(box['label'])

  if output == 'print':
    for label in labels:
      print(label)
  elif output == 'return':
    return labels
  else:
    raise Exception("find_all_classes: invalid output type %s" % repr(output))

# Prints command line usage for all actions or for a specific action
def usage(action=None):
  known_actions = ['find-all-classes']
  # if user is trying to use an unknown action, print usage for all actions
  if action != None and not action in known_actions:
    print("Unknown action %s\n" % repr(action))
    action = None

  if action == None:
    print("Usage: %s <action> [params...]\n" % sys.argv[0])
  else:
    print("Usage for action '%s':" % action)

  if action == None or action == 'find-all-classes':
    print("%s find-all-classes <bstld_labels_file.yaml>" % sys.argv[0])
    print("Finds all unique classes in a BSTLD-formatted labels file and prints them.\n")

if __name__ == "__main__":
  # parse command line arguments
  argv_len = len(sys.argv)
  if argv_len == 0:
    # should never happen
    pass
  elif argv_len == 1:
    # show usage
    usage()
  else:
    # run action or show usage for bad params
    action = sys.argv[1]

    if action == 'find-all-classes':
      if argv_len == 3:
        labels_file = sys.argv[2]
        find_all_classes(labels_file, 'print')
      else:
        usage(action)
    else:
      # unknown action
      usage(action)
