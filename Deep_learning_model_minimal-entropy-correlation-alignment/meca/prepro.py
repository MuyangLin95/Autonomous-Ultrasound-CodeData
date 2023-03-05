import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def main():
    size_x, size_y = 50, 50
    datadirs = [f"dataset/dataset_{i}/" for i in range(10)]
    ends = ['CA', 'nCA']
    for (i, datadir) in enumerate(datadirs):
        images_array = np.zeros((0,size_x,size_y,1))
        labels = []
        for end in ends:
            for name in os.listdir(datadir+end):
                img = Image.open(os.path.join(datadir+end, name))
                img = img.Resize((size_x,size_y))
                img = np.asarray(img)
                images_array = np.concatenate((images_array,img.reshape(1,size_x,size_y,1)))
                labels.append(0 if end == 'nCA' else 1)
        Xtr, Xva, Ytr, Yva = train_test_split(images_array,labels,test_size=0.25)
        np.save(f"tds/{i+1}.npy",Xtr)
        np.save(f"tds/{i+1}_Y.npy",Ytr)
        np.save(f"tds/va_{i+1}.npy",Xva)
        np.save(f"tds/va_{i+1}_Y.npy",Yva)

if __name__ == "__main__":
    main()
    
