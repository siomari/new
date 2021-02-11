import os 

def data_preprocess(images_path, masks_path):

    _, _, image_files = next(os.walk(images_path))

    _, _, mask_files = next(os.walk(masks_path))

    
    for i in image_files:
        if i not in mask_files:
            image_files.remove(i)
            
    for i in mask_files:
        if i not in image_files:
            mask_files.remove(i)


    for idx, i in enumerate(image_files):
        image_files[idx] = os.path.abspath(os.path.join(images_path, i))
    
    for idx, i in enumerate(mask_files):
        mask_files[idx] = os.path.abspath(os.path.join(masks_path, i))


    return image_files, mask_files


