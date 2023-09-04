def CropToMatch(input, shortcut):
    diff = max(0, input.shape[2] - shortcut.shape[2])
    start = diff // 2 + diff % 2

    return input[:, :, start:start+shortcut.shape[2]]
   