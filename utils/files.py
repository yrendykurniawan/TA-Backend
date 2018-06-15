import os


dirname=os.path.dirname
MAIN_DIRECTORY = dirname(dirname(__file__))

def get_full_path(*path):
    foldernya = os.path.join(MAIN_DIRECTORY, *path)
    foldernya = foldernya.replace("\\", "/")
    return foldernya

#path_to_map = get_full_path('res', 'asd.png')
#print(path_to_map)