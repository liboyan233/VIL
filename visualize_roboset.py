from roboset_dataloader import RoboSetDataset

path = '/home/bohanfeng/Desktop/liboyan/Imitation_learning/dataset/RoboSet/baking_prep/Slide_Open_Drawer/scene_1/baking_prep_slide_open_drawer_scene_1/'
filename = 'training_set/baking_prep_open_drawer_scene_1_20230307-144708.h5'
data = RoboSetDataset.datareader_all(path+filename) # N, 240, 424, 3
imgs = data['rgb_top'][()]
for i in range(imgs.shape[0]):
    RoboSetDataset.visualize(imgs[i])
print(data.keys())