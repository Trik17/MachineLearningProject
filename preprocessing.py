def removeNullTargetRows(panda,target):
    return panda[panda[target].notna()]
