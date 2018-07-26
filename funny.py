import os
from RMDL import text_feature_extraction as txt
from sklearn.model_selection import train_test_split
from RMDL.Download import Download_WOS as WOS
import numpy as np
from RMDL import RMDL_Text as RMDL

if __name__ == "__main__":
    path_WOS = '../dataset/haha_single_video_item/haha_single_video_item.raw.20180723'
    fname = os.path.join(path_WOS + '.X')
    fnamek = os.path.join(path_WOS + '.Y1')
    with open(fname, encoding="utf-8") as f:
        content = f.readlines()
        content = [txt.text_cleaner(x) for x in content]
    with open(fnamek) as fk:
        contentk = fk.readlines()
    contentk = [x.strip() for x in contentk]
    Label = np.matrix(contentk, dtype=int)
    Label = np.transpose(Label)
    np.random.seed(7)
    print(Label.shape)
    X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=0, shuffle=False)

    batch_size = 128
    n_epochs = [0, 2, 2]  ## DNN--RNN-CNN
    Random_Deep = [0, 1, 1]  ## DNN--RNN-CNN

    RMDL.Text_Classification(X_train, y_train, X_test, y_test,
                             batch_size=batch_size,
                             sparse_categorical=True,
                             random_deep=Random_Deep,
                             epochs=n_epochs,
                             GloVe_dir="../dataset/",
                             GloVe_file="glove.6B.300d.txt",
                             EMBEDDING_DIM=300,
                             MAX_SEQUENCE_LENGTH=100,
                             MAX_NB_WORDS=50000)
