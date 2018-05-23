from glob import glob
from age_feature import *
import os

data_path = '../DataSet/UTKFace-align-train'
file_names = glob(os.path.join(data_path,'*.jpg'))
size_data = len(file_names)

a_feautre = learn_age_feature('config.yml')
diff_age = np.zeros([size_data,2])   # store id,diff of age

img_info_ori = [[0]*2 for _ in range(size_data)]  # id, img name
img_info = [[0]*2 for _ in range(size_data)]    # ordered id, img name

pred = np.zeros(shape=[size_data])
realage = np.zeros(shape=[size_data])

for i,i_img in enumerate(file_names):
    try:
        dataname = str(i_img).split('/')[2]
    except:
        dataname = i_img.encode('ascii', 'ignore').decode('ascii')

    filename = str(i_img).split('/')[3]
    realage[i] = str(filename).split('_')[0]
    img = cv2.imread(i_img)   
    face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    for c_i in range(3):
        face[:, :, c_i] -= a_feautre.means[c_i]

    face = cv2.resize(face, tuple(a_feautre.size))
    data = np.expand_dims(face.transpose(2, 0, 1), 0)

    b = a_feautre.net.forward_all(data=data);

    age_feature = a_feautre.net.blobs['fc8-101'].data


    feature_dir = data_path+'-101-feature/'+filename+'.npy'
    np.save(feature_dir, age_feature)  #save age feature

    pred[i] = np.argmax(b['prob'])  # estimated age
    realage[i] = str(filename).split('_')[0]
    diff_age[i,0] = i
    diff_age[i,1] = np.abs(pred[i]-realage[i])
    img_info_ori[i][0] = str(i)
    img_info_ori[i][1] = i_img

# sort result by diff ascent
ind = np.argsort(diff_age[:,1])
diff_age = diff_age[ind]
#index images by diff ascent order
for i,_ in enumerate(ind):
    j = ind[i]
    img_info[i][:] = img_info_ori[j][:]

age_dis = np.subtract(pred,realage)
np.save('diffage-UTKFace.npy',diff_age)
np.save('imginfo-UTKFace.npy',img_info)
print 'size of data:  ',size_data
print 'var_age:   ',np.var(age_dis)
print 'mae_age:   ',np.sum(np.abs(age_dis))/size_data
