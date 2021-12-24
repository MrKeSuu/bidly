SOURCE_FILEPATH = 'obj.names'


def main():
    with open(SOURCE_FILEPATH, 'r') as fi:
        for id_, name in enumerate(fi):
            print('ID:', id_, '\tName:', name)

if __name__ == '__main__':
    main()
