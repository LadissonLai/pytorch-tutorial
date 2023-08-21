from torch.utils.data import Dataset
import os
import cv2

# 总结：继承Dataset类，必须实现__len__和__getitem__方法。
# 这两个方法对应类实例化对象的len()和[]方法


class MyDataset(Dataset):
    def __init__(self, data_path, label):
        self.data_path = data_path
        self.label = label
        self.image_path = os.path.join(self.data_path, self.label)
        self.image_items = os.listdir(self.image_path)

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        image_idx = self.image_items[idx]
        image_abs_path = os.path.join(self.image_path, image_idx)
        image = cv2.imread(image_abs_path)
        return image, self.label


if __name__ == "__main__":
    data_dir = "../dataset/train"
    data_label = "ants"
    my_dataset = MyDataset(data_dir, data_label)
    print(len(my_dataset))
    img, label = my_dataset[0]
    cv2.imshow(label, img)
    key = cv2.waitKey(0)
    if key == ord("q"):
        cv2.destroyAllWindows()
