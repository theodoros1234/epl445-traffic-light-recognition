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


# Extracts wanted command line options
def extract_cmd_args(action, args, wanted_options):
  extracted = dict()
  other = list()

  # go through args
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith('-'):                     # is an option
      if arg in wanted_options:                 # is an option we want
        if wanted_options[arg]['gets_param']:   # option gets param
          if i + 1 < len(args):
            extracted[arg] = args[i + 1]
            i += 1
          else:                                 # missing param
            print("Missing parameter for option %s\n" % repr(arg))
            usage(action)
            return
        else:                                   # option doesn't get param
          extracted[arg] = True
      else:                                     # is NOT an option we want
        print("Unknown option %s\n" % repr(arg))
        usage(action)
        return

    else: # not an option
      other.append(arg)

    i += 1

  # Make sure required options were included
  for option_name in wanted_options:
    option = wanted_options[option_name]
    if option['required'] and not option_name in extracted:
      print("Missing required option %s\n" % repr(option_name))
      usage()
      return

  return (extracted, other)


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
