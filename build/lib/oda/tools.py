from pathlib import Path



def getfiles(dir_data, characteristic_str='*nc'):
    '''
    files = getfiles(dir_data, characteristic_str='*nc'):
    Input
        ::dir_data: str, default is pwd()
        characteristic_str: str, default is '*nc'
    Output:
        files list.
    '''
    var_path = Path(dir_data)
    files = list(var_path.glob('**/'+characteristic_str))
    files.sort()
    return files

