import pickle
import matplotlib.pyplot as plt
import numpy as np

class DataGenerate():
    def __init__(self,path,mode):
        # WAITING FOR CODE PACKAGE TO SYNC UP
        with open(path, mode) as f:
            self.data = pickle.load(f)
        self.X_train, self.y_train = self.data['features'], self.data['labels']
        print("Number of training examples =", self.X_train.shape[0])
        print("Image data shape =", self.X_train[0].shape)
        print("Number of classes =", len(set(self.y_train)))
    
    def checkInitData(self):
        n_classes = len(set(self.y_train))
        n_data = self.X_train.shape[0]
        rows,cols=4,12
        fig,ax_array = plt.subplots(rows,cols)
        plt.suptitle('RANDOM SAMPLES FROM TRAINING SET (one for each class)')
        for classIndex,ax in enumerate(ax_array.ravel()):
            if classIndex < n_classes:
                cur_X = self.X_train[self.y_train == classIndex]
                cur_img = cur_X[np.random.randint(len(cur_X))]
                ax.imshow(cur_img)
                ax.set_title('{:02d}'.format(classIndex))
            else:
                ax.axis('off')
        plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
        plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
        plt.draw()
        

        data_distribution = np.zeros(n_classes)
        for c_rate in range(n_classes):
            data_distribution[c_rate] = np.sum(self.y_train == c_rate)/n_data
        fig_dis,ax_dis=plt.subplots()
        col_width = 0.5
        bar_data = ax_dis.bar(np.arange(n_classes)+col_width, data_distribution, width=col_width, color='b')
        ax_dis.set_ylabel('PERCENTAGE OF PRESENCE')
        ax_dis.set_xlabel('CLASS LABEL')
        ax_dis.set_title('Classes distribution in traffic-sign dataset')
        ax_dis.set_xticks(np.arange(0, n_classes, 5) )
        ax_dis.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
        #ax_dis.legend((bar_data[0]), ('data set'))
        plt.draw()
        plt.show()


littleData=DataGenerate(path='train.p', mode='rb')
littleData.checkInitData()





