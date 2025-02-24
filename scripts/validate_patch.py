import json
import sys

if __name__ == "__main__":
    print(sys.argv)
    with open(sys.argv[1]) as fin:
        lines_changed = json.load(fin)
        print(type(lines_changed))
        print(lines_changed)
